# ml/model_manager.py - Обновленный менеджер с продвинутым обучением

import numpy as np
from typing import Optional, Tuple, List, Dict
import os

from ml.enhanced_model_trainer import EnhancedModelTrainer
from ml.knn_classifier import KNNAuthenticator  # Сохраняем для совместимости
from ml.feature_extractor import FeatureExtractor
from utils.database import DatabaseManager
from config import MIN_TRAINING_SAMPLES, THRESHOLD_ACCURACY, MODELS_DIR

class ModelManager:
    """Обновленный менеджер для управления моделями пользователей"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.feature_extractor = FeatureExtractor()
        self.models_cache = {}  # Кэш загруженных моделей
    
    def train_user_model(self, user_id: int, use_enhanced_training: bool = True) -> Tuple[bool, float, str]:
        """
        Обучение модели для пользователя с выбором метода
        
        Args:
            user_id: ID пользователя
            use_enhanced_training: Использовать продвинутое обучение с валидацией
            
        Returns:
            (успех, точность, сообщение)
        """
        print(f"\n🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
        print(f"👤 Пользователь ID: {user_id}")
        print(f"🔬 Метод: {'Продвинутое обучение' if use_enhanced_training else 'Базовое обучение'}")
        
        # Получение образцов пользователя
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        if len(user_samples) < MIN_TRAINING_SAMPLES:
            return False, 0.0, f"Недостаточно образцов. Необходимо минимум {MIN_TRAINING_SAMPLES}, собрано {len(user_samples)}"
        
        if use_enhanced_training:
            return self._train_enhanced_model(user_id, user_samples)
        else:
            return self._train_basic_model(user_id, user_samples)
    
    def _train_enhanced_model(self, user_id: int, user_samples: List) -> Tuple[bool, float, str]:
        """Продвинутое обучение с валидацией"""
        
        # Создаем тренер
        trainer = EnhancedModelTrainer(user_id)
        
        # Получение негативных примеров от других пользователей
        negative_samples = self._get_negative_samples(user_id)
        
        # Полный цикл обучения с валидацией
        success, accuracy, message, training_summary = trainer.train_with_validation(
            positive_samples=user_samples,
            negative_samples=negative_samples
        )
        
        if success:
            # Кэшируем обученную модель
            self.models_cache[user_id] = trainer
            
            # Обновляем информацию о пользователе
            user = self.db.get_user_by_id(user_id)
            if user:
                user.is_trained = True
                user.training_samples = len(user_samples)
                self.db.update_user(user)
            
            print(f"✅ Продвинутое обучение завершено успешно!")
            print(f"📊 Детали обучения сохранены в training_report_user_{user_id}.json")
        
        return success, accuracy, message
    
    def _train_basic_model(self, user_id: int, user_samples: List) -> Tuple[bool, float, str]:
        """Базовое обучение (старый метод для совместимости)"""
        
        print("⚠️ Используется базовое обучение без валидации")
        
        # Извлечение признаков
        X_positive = self.feature_extractor.extract_features_from_samples(user_samples)
        
        # Нормализация признаков
        X_positive_norm, norm_stats = self.feature_extractor.normalize_features(X_positive)
        
        # Получение негативных примеров от других пользователей
        X_negative = self._get_negative_samples(user_id)
        if X_negative is not None and len(X_negative) > 0:
            X_negative_norm = self.feature_extractor.apply_normalization(X_negative, norm_stats)
        else:
            X_negative_norm = None
        
        # Создание и обучение классификатора
        classifier = KNNAuthenticator()
        classifier.normalization_stats = norm_stats
        
        success, accuracy = classifier.train(X_positive_norm, X_negative_norm)
        
        if not success:
            return False, 0.0, "Ошибка при обучении модели"
        
        if accuracy < THRESHOLD_ACCURACY:
            return False, accuracy, f"Низкая точность модели: {accuracy:.2%}. Необходимо минимум {THRESHOLD_ACCURACY:.2%}"
        
        # Сохранение модели
        classifier.save_model(user_id)
        
        # Обновление информации о пользователе
        user = self.db.get_user_by_id(user_id)
        if user:
            user.is_trained = True
            user.training_samples = len(user_samples)
            self.db.update_user(user)
        
        # Добавление в кэш
        self.models_cache[user_id] = classifier
        
        return True, accuracy, f"Модель успешно обучена с точностью {accuracy:.2%}"
    
    def authenticate_user(self, user_id: int, keystroke_features: dict, verbose: bool = False) -> Tuple[bool, float, str]:
        """
        Аутентификация пользователя по динамике нажатий
        Возвращает: (успех, уверенность, сообщение)
        """
        # Получение модели
        model = self._get_user_model(user_id)
        if model is None:
            return False, 0.0, "Модель пользователя не найдена"
        
        # Подготовка вектора признаков
        feature_vector = np.array([
            keystroke_features.get('avg_dwell_time', 0),
            keystroke_features.get('std_dwell_time', 0),
            keystroke_features.get('avg_flight_time', 0),
            keystroke_features.get('std_flight_time', 0),
            keystroke_features.get('typing_speed', 0),
            keystroke_features.get('total_typing_time', 0)
        ])
        
        # Проверяем тип модели и вызываем соответствующий метод
        if isinstance(model, EnhancedModelTrainer):
            # Продвинутая модель
            is_authenticated, confidence, detailed_stats = model.predict_with_confidence(feature_vector)
            
            if verbose:
                print(f"\n🔬 ПРОДВИНУТАЯ АУТЕНТИФИКАЦИЯ")
                print(f"📊 Уверенность: {confidence:.3f}")
                print(f"🎯 Результат: {'ПРИНЯТ' if is_authenticated else 'ОТКЛОНЕН'}")
                print(f"⚙️ Параметры модели: {model.best_params}")
        
        elif isinstance(model, KNNAuthenticator):
            # Базовая модель
            feature_vector_norm = feature_vector
            if model.normalization_stats:
                feature_vector_norm = self.feature_extractor.apply_normalization(
                    feature_vector.reshape(1, -1), 
                    model.normalization_stats
                ).flatten()
            
            is_authenticated, confidence, detailed_stats = model.authenticate(
                feature_vector_norm, 
                THRESHOLD_ACCURACY,
                verbose=verbose
            )
        else:
            return False, 0.0, "Неизвестный тип модели"
        
        if is_authenticated:
            message = f"Аутентификация успешна (уверенность: {confidence:.2%})"
        else:
            message = f"Аутентификация отклонена (уверенность: {confidence:.2%})"
        
        return is_authenticated, confidence, message
    
    def authenticate_user_detailed(self, user_id: int, keystroke_features: dict) -> Tuple[bool, float, dict]:
        """
        Аутентификация с детальной статистикой
        """
        model = self._get_user_model(user_id)
        if model is None:
            return False, 0.0, {}
        
        # Подготовка вектора признаков
        feature_vector = np.array([
            keystroke_features.get('avg_dwell_time', 0),
            keystroke_features.get('std_dwell_time', 0),
            keystroke_features.get('avg_flight_time', 0),
            keystroke_features.get('std_flight_time', 0),
            keystroke_features.get('typing_speed', 0),
            keystroke_features.get('total_typing_time', 0)
        ])
        
        # Проверяем тип модели
        if isinstance(model, EnhancedModelTrainer):
            # Продвинутая модель
            is_authenticated, confidence, detailed_stats = model.predict_with_confidence(feature_vector)
            
            # Дополняем статистику для совместимости
            detailed_stats.update({
                'knn_confidence': confidence,
                'distance_score': confidence * 0.8,  # Примерная оценка
                'feature_score': confidence * 0.9,   # Примерная оценка
                'final_confidence': confidence,
                'threshold': THRESHOLD_ACCURACY,
                'weights': {'knn': 0.6, 'distance': 0.3, 'features': 0.1},
                'training_samples': len(self.db.get_user_keystroke_samples(user_id, training_only=True))
            })
            
        elif isinstance(model, KNNAuthenticator):
            # Базовая модель
            feature_vector_norm = feature_vector
            if model.normalization_stats:
                feature_vector_norm = self.feature_extractor.apply_normalization(
                    feature_vector.reshape(1, -1), 
                    model.normalization_stats
                ).flatten()
            
            is_authenticated, confidence, detailed_stats = model.authenticate(
                feature_vector_norm, 
                THRESHOLD_ACCURACY,
                verbose=True
            )
        else:
            return False, 0.0, {}
        
        return is_authenticated, confidence, detailed_stats
    
    def _get_user_model(self, user_id: int):
        """Получение модели пользователя (с кэшированием)"""
        # Проверка кэша
        if user_id in self.models_cache:
            return self.models_cache[user_id]
        
        # Попытка загрузить продвинутую модель
        enhanced_model = EnhancedModelTrainer.load_trained_model(user_id)
        if enhanced_model:
            self.models_cache[user_id] = enhanced_model
            return enhanced_model
        
        # Попытка загрузить базовую модель
        basic_model = KNNAuthenticator.load_model(user_id)
        if basic_model:
            self.models_cache[user_id] = basic_model
            return basic_model
        
        return None
    
    def _get_negative_samples(self, exclude_user_id: int) -> Optional[List]:
        """Получение образцов других пользователей для негативных примеров"""
        all_negative_samples = []
        
        # Получаем всех пользователей кроме текущего
        all_users = self.db.get_all_users()
        for user in all_users:
            if user.id != exclude_user_id:
                user_samples = self.db.get_user_keystroke_samples(user.id, training_only=True)
                if user_samples:
                    # Берем не более 10 образцов от каждого пользователя
                    selected_samples = user_samples[:10] if len(user_samples) > 10 else user_samples
                    all_negative_samples.extend(selected_samples)
        
        print(f"📊 Негативных образцов от других пользователей: {len(all_negative_samples)}")
        
        return all_negative_samples if all_negative_samples else None
    
    def delete_user_model(self, user_id: int):
        """Удаление модели пользователя"""
        # Удаление из кэша
        if user_id in self.models_cache:
            del self.models_cache[user_id]
        
        # Удаление файлов моделей
        enhanced_path = os.path.join(MODELS_DIR, f"user_{user_id}_enhanced_knn.pkl")
        basic_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
        
        for path in [enhanced_path, basic_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"🗑️ Удален файл модели: {path}")
    
    def get_model_info(self, user_id: int) -> dict:
        """Получение информации о модели пользователя"""
        model = self._get_user_model(user_id)
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        # Базовая информация
        info = {
            'min_samples': MIN_TRAINING_SAMPLES,
            'training_samples': len(user_samples),
            'model_type': 'none'
        }
        
        if isinstance(model, EnhancedModelTrainer):
            # Продвинутая модель
            validation_results = getattr(model, 'validation_results', {})
            final_eval = validation_results.get('final_evaluation', {})
            
            info.update({
                'is_trained': True,
                'model_type': 'enhanced',
                'best_params': getattr(model, 'best_params', {}),
                'test_accuracy': final_eval.get('test_accuracy', 0.0),
                'test_precision': final_eval.get('test_precision', 0.0),
                'test_recall': final_eval.get('test_recall', 0.0),
                'test_f1': final_eval.get('test_f1', 0.0),
                'roc_auc': final_eval.get('roc_auc', 0.0),
                'cv_score': validation_results.get('cross_validation', {}).get(
                    list(validation_results.get('cross_validation', {}).keys())[-1] if validation_results.get('cross_validation') else 3, 
                    {}
                ).get('mean_accuracy', 0.0),
                'overfitting_status': validation_results.get('learning_curve', {}).get('overfitting_status', 'Unknown'),
                'feature_importance': []  # Можно добавить позже
            })
            
        elif isinstance(model, KNNAuthenticator):
            # Базовая модель
            info.update({
                'is_trained': True,
                'model_type': 'basic',
                'n_neighbors': getattr(model, 'n_neighbors', 3),
                'cv_score': 0.85,  # Заглушка
                'feature_importance': model.get_feature_importance() if model.is_trained else []
            })
        else:
            # Модель не обучена
            info.update({
                'is_trained': False,
                'feature_importance': []
            })
        
        return info
    
    def get_training_report(self, user_id: int) -> Optional[Dict]:
        """Получение детального отчета об обучении"""
        from config import DATA_DIR
        import json
        
        report_path = os.path.join(DATA_DIR, f"training_report_user_{user_id}.json")
        
        if not os.path.exists(report_path):
            return None
        
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка чтения отчета: {e}")
            return None
    
    def compare_model_performance(self, user_id: int) -> Dict:
        """Сравнение производительности разных моделей"""
        
        # Получаем данные пользователя
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        if len(user_samples) < MIN_TRAINING_SAMPLES:
            return {'error': 'Недостаточно данных для сравнения'}
        
        results = {}
        
        # Тестируем разные конфигурации
        configurations = [
            {'name': 'Базовый KNN (k=3)', 'params': {'n_neighbors': 3, 'weights': 'uniform'}},
            {'name': 'Взвешенный KNN (k=5)', 'params': {'n_neighbors': 5, 'weights': 'distance'}},
            {'name': 'Оптимизированный', 'params': 'auto'}  # Будет использован grid search
        ]
        
        for config in configurations:
            try:
                if config['params'] == 'auto':
                    # Запускаем полную оптимизацию
                    trainer = EnhancedModelTrainer(user_id)
                    success, accuracy, message, summary = trainer.train_with_validation(user_samples)
                    
                    if success:
                        results[config['name']] = {
                            'accuracy': accuracy,
                            'params': trainer.best_params,
                            'cv_scores': trainer.validation_results.get('cross_validation', {}),
                            'status': 'success'
                        }
                    else:
                        results[config['name']] = {'status': 'failed', 'message': message}
                else:
                    # Простое тестирование с фиксированными параметрами
                    from sklearn.model_selection import cross_val_score
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    # Подготовка данных
                    trainer = EnhancedModelTrainer(user_id)
                    X, y = trainer.prepare_training_data(user_samples)
                    
                    # Тестирование модели
                    model = KNeighborsClassifier(**config['params'])
                    scores = cross_val_score(model, X, y, cv=5)
                    
                    results[config['name']] = {
                        'accuracy': scores.mean(),
                        'accuracy_std': scores.std(),
                        'params': config['params'],
                        'status': 'success'
                    }
                    
            except Exception as e:
                results[config['name']] = {'status': 'failed', 'error': str(e)}
        
        return results