# ml/enhanced_model_trainer.py - Исправленная продвинутая система обучения

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, validation_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
import json

from ml.feature_extractor import FeatureExtractor
from config import MODELS_DIR, DATA_DIR

class EnhancedModelTrainer:
    """Продвинутая система обучения с валидацией и оптимизацией"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
        # Результаты валидации
        self.validation_results = {}
        self.best_model = None
        self.best_params = {}
        self.training_history = []
    
    def prepare_training_data(self, positive_samples: List, negative_samples: List = None) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения с улучшенной генерацией негативов"""
        
        print(f"\n🔬 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        
        # Извлечение признаков из положительных образцов
        X_positive = self.feature_extractor.extract_features_from_samples(positive_samples)
        n_positive = len(X_positive)
        
        print(f"✅ Положительных образцов: {n_positive}")
        
        # Генерация или использование негативных образцов
        if negative_samples is None or len(negative_samples) == 0:
            X_negative = self._generate_enhanced_negatives(X_positive)
        else:
            X_negative = self.feature_extractor.extract_features_from_samples(negative_samples)
        
        n_negative = len(X_negative)
        print(f"✅ Негативных образцов: {n_negative}")
        
        # Комбинирование данных
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
        
        # Нормализация признаков
        X_normalized = self.scaler.fit_transform(X)
        
        print(f"📊 Итоговый датасет: {len(X)} образцов, {X.shape[1]} признаков")
        print(f"📈 Баланс классов: {n_positive/(n_positive+n_negative)*100:.1f}% положительных")
        
        return X_normalized, y
    
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict:
        """Кросс-валидация для оценки качества модели"""
        
        print(f"\n🔄 КРОСС-ВАЛИДАЦИЯ ({cv_folds} fold)")
        
        # Стратифицированная кросс-валидация (сохраняет пропорции классов)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Тестируем разные количества соседей
        k_range = range(1, min(15, len(X)//4))
        cv_results = {}
        
        for k in k_range:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights='distance',
                metric='euclidean',
                algorithm='ball_tree'
            )
            
            # Кросс-валидация
            scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
            
            cv_results[k] = {
                'mean_accuracy': float(scores.mean()),  # ✅ Конвертируем в float
                'std_accuracy': float(scores.std()),    # ✅ Конвертируем в float
                'scores': scores.tolist()               # ✅ Конвертируем в list
            }
            
            print(f"K={k:2d}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Находим оптимальное K
        best_k = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_accuracy'])
        
        print(f"\n🎯 Лучшее K: {best_k} (точность: {cv_results[best_k]['mean_accuracy']:.3f})")
        
        self.validation_results['cross_validation'] = cv_results
        self.best_params['n_neighbors'] = best_k
        
        return cv_results
    
    def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Оптимизация гиперпараметров через Grid Search"""
        
        print(f"\n⚙️ ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ")
        
        # Параметры для поиска
        param_grid = {
            'n_neighbors': range(1, min(15, len(X)//3)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
        
        # Grid Search с кросс-валидацией
        knn = KNeighborsClassifier()
        
        grid_search = GridSearchCV(
            knn, param_grid, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,  # Используем все ядра процессора
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"🏆 Лучшие параметры: {grid_search.best_params_}")
        print(f"🎯 Лучшая точность: {grid_search.best_score_:.3f}")
        
        self.best_params.update(grid_search.best_params_)
        
        # ✅ ИСПРАВЛЕНИЕ: Конвертируем результаты в JSON-совместимые типы
        cv_results_serializable = {}
        for key, value in grid_search.cv_results_.items():
            if isinstance(value, np.ndarray):
                cv_results_serializable[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                cv_results_serializable[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                cv_results_serializable[key] = float(value)
            else:
                cv_results_serializable[key] = value
        
        self.validation_results['grid_search'] = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),  # ✅ Конвертируем в float
            'cv_results': cv_results_serializable          # ✅ Используем очищенные результаты
        }
        
        return grid_search.best_params_
    
    def learning_curve_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Анализ кривых обучения для оценки переобучения"""
        
        print(f"\n📈 АНАЛИЗ КРИВЫХ ОБУЧЕНИЯ")
        
        from sklearn.model_selection import learning_curve
        
        # Создаем модель с лучшими параметрами
        best_knn = KNeighborsClassifier(**self.best_params)
        
        # Размеры обучающих выборок
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            best_knn, X, y,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Статистики
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # ✅ ИСПРАВЛЕНИЕ: Конвертируем numpy массивы в списки
        learning_results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist()
        }
        
        # Анализ переобучения
        final_gap = float(train_mean[-1] - val_mean[-1])  # ✅ Конвертируем в float
        if final_gap > 0.1:
            overfitting_status = "ВЫСОКИЙ риск переобучения"
        elif final_gap > 0.05:
            overfitting_status = "СРЕДНИЙ риск переобучения"
        else:
            overfitting_status = "НИЗКИЙ риск переобучения"
        
        print(f"📊 Разрыв обучение/валидация: {final_gap:.3f}")
        print(f"🎯 Статус: {overfitting_status}")
        
        learning_results['overfitting_gap'] = final_gap
        learning_results['overfitting_status'] = overfitting_status
        
        self.validation_results['learning_curve'] = learning_results
        
        return learning_results
    
    def detailed_evaluation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Детальная оценка финальной модели"""
        
        print(f"\n🎯 ДЕТАЛЬНАЯ ОЦЕНКА МОДЕЛИ")
        
        # Разделение на обучение и тест
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Обучение финальной модели
        final_model = KNeighborsClassifier(**self.best_params)
        final_model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]
        
        # Метрики
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # ✅ ИСПРАВЛЕНИЕ: Конвертируем все метрики в стандартные типы Python
        evaluation_results = {
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_precision': float(precision_score(y_test, y_pred)),
            'test_recall': float(recall_score(y_test, y_pred)),
            'test_f1': float(f1_score(y_test, y_pred)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()  # ✅ Конвертируем в list
        }
        
        # ROC-AUC если есть вероятности
        if len(np.unique(y_test)) > 1:
            evaluation_results['roc_auc'] = float(roc_auc_score(y_test, y_prob))
            
            # ROC кривая
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            evaluation_results['roc_curve'] = {
                'fpr': fpr.tolist(),  # ✅ Конвертируем в list
                'tpr': tpr.tolist()   # ✅ Конвертируем в list
            }
        
        print(f"🎯 Точность на тесте: {evaluation_results['test_accuracy']:.3f}")
        print(f"🎯 Precision: {evaluation_results['test_precision']:.3f}")
        print(f"🎯 Recall: {evaluation_results['test_recall']:.3f}")
        print(f"🎯 F1-score: {evaluation_results['test_f1']:.3f}")
        if 'roc_auc' in evaluation_results:
            print(f"🎯 ROC-AUC: {evaluation_results['roc_auc']:.3f}")
        
        self.validation_results['final_evaluation'] = evaluation_results
        self.best_model = final_model
        
        return evaluation_results
    
    def train_with_validation(self, positive_samples: List, negative_samples: List = None) -> Tuple[bool, float, str, Dict]:
        """Полный цикл обучения с валидацией"""
        
        training_start = datetime.now()
        
        print(f"\n🚀 ЗАПУСК ПРОДВИНУТОГО ОБУЧЕНИЯ")
        print(f"⏰ Время начала: {training_start.strftime('%H:%M:%S')}")
        print(f"👤 Пользователь ID: {self.user_id}")
        
        try:
            # 1. Подготовка данных
            X, y = self.prepare_training_data(positive_samples, negative_samples)
            
            # Проверка минимального количества данных
            if len(X) < 20:
                return False, 0.0, "Недостаточно данных для качественного обучения (минимум 20 образцов)", {}
            
            # 2. Кросс-валидация
            cv_results = self.perform_cross_validation(X, y)
            
            # 3. Оптимизация гиперпараметров
            if len(X) >= 30:  # Только если достаточно данных
                best_params = self.hyperparameter_optimization(X, y)
            else:
                print("⚠️ Недостаточно данных для Grid Search, используем простую валидацию")
            
            # 4. Анализ кривых обучения
            learning_results = self.learning_curve_analysis(X, y)
            
            # 5. Финальная оценка
            evaluation_results = self.detailed_evaluation(X, y)
            
            # 6. Сохранение результатов
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # ✅ ИСПРАВЛЕНИЕ: Убеждаемся, что все данные JSON-совместимы
            training_summary = {
                'user_id': int(self.user_id),  # ✅ Убеждаемся что это int
                'training_start': training_start.isoformat(),
                'training_end': training_end.isoformat(),
                'training_duration': float(training_duration),  # ✅ Конвертируем в float
                'dataset_size': int(len(X)),                    # ✅ Конвертируем в int
                'n_positive': int(np.sum(y)),                   # ✅ Конвертируем в int
                'n_negative': int(len(y) - np.sum(y)),          # ✅ Конвертируем в int
                'best_params': self.best_params,
                'validation_results': self.validation_results
            }
            
            self.training_history.append(training_summary)
            
            # Сохранение модели и результатов
            self._save_training_results(training_summary)
            
            # Финальный отчет
            final_accuracy = evaluation_results['test_accuracy']
            
            print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print(f"⏱️ Время обучения: {training_duration:.1f} секунд")
            print(f"🎯 Финальная точность: {final_accuracy:.3f}")
            print(f"🏆 Лучшие параметры: {self.best_params}")
            
            # Определяем качество модели
            if final_accuracy >= 0.9:
                quality_message = "Отличная модель!"
            elif final_accuracy >= 0.8:
                quality_message = "Хорошая модель"
            elif final_accuracy >= 0.7:
                quality_message = "Приемлемая модель"
            else:
                quality_message = "Слабая модель, рекомендуется больше данных"
            
            success_message = f"Обучение завершено успешно. {quality_message} (точность: {final_accuracy:.1%})"
            
            return True, final_accuracy, success_message, training_summary
            
        except Exception as e:
            error_message = f"Ошибка при обучении: {str(e)}"
            print(f"❌ {error_message}")
            import traceback
            traceback.print_exc()
            return False, 0.0, error_message, {}
    
    def _generate_enhanced_negatives(self, X_positive: np.ndarray) -> np.ndarray:
        """Улучшенная генерация негативных примеров с разнообразными стратегиями"""
        
        n_samples = len(X_positive)
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
        
        # Обеспечиваем минимальную вариативность
        std = np.maximum(std, mean * 0.1)
        
        negatives = []
        
        # Стратегия 1: Статистические выбросы (25%)
        outlier_count = n_samples // 4
        for i in range(outlier_count):
            # Генерируем выбросы на расстоянии 2-4 стандартных отклонений
            direction = np.random.choice([-1, 1], size=len(mean))
            magnitude = np.random.uniform(2, 4)
            sample = mean + direction * magnitude * std
            sample = np.maximum(sample, mean * 0.01)  # Избегаем отрицательных значений
            negatives.append(sample)
        
        # Стратегия 2: Противоположные паттерны (30%)
        opposite_count = int(n_samples * 0.3)
        for i in range(opposite_count):
            # Инвертируем некоторые признаки
            sample = mean.copy()
            features_to_invert = np.random.choice(len(mean), size=np.random.randint(2, 4), replace=False)
            
            for feat_idx in features_to_invert:
                if feat_idx in [0, 2]:  # времена
                    # Очень быстрое vs очень медленное
                    factor = np.random.choice([0.2, 4.0])
                elif feat_idx in [1, 3]:  # вариативности
                    # Очень стабильное vs очень нестабильное
                    factor = np.random.choice([0.1, 3.0])
                elif feat_idx == 4:  # скорость
                    # Очень медленно vs очень быстро
                    factor = np.random.choice([0.3, 3.0])
                else:  # общее время
                    factor = np.random.choice([0.4, 2.5])
                
                sample[feat_idx] = mean[feat_idx] * factor
            
            negatives.append(sample)
        
        # Стратегия 3: Шумовые вариации (25%)
        noise_count = n_samples // 4
        for i in range(noise_count):
            # Добавляем различные типы шума
            noise_type = np.random.choice(['gaussian', 'uniform', 'exponential'])
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, std * 2)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-std * 3, std * 3)
            else:  # exponential
                noise = np.random.exponential(std) * np.random.choice([-1, 1], size=len(std))
            
            sample = mean + noise
            sample = np.maximum(sample, mean * 0.05)
            negatives.append(sample)
        
        # Стратегия 4: Межклассовые границы (20%)
        boundary_count = n_samples - outlier_count - opposite_count - noise_count
        for i in range(boundary_count):
            # Генерируем образцы близко к границе решения
            # Используем случайные линейные комбинации обучающих примеров с добавлением шума
            
            # Выбираем 2-3 случайных обучающих примера
            indices = np.random.choice(len(X_positive), size=np.random.randint(2, 4), replace=False)
            weights = np.random.dirichlet(np.ones(len(indices)))  # Случайные веса, сумма = 1
            
            # Создаем комбинацию
            sample = np.zeros_like(mean)
            for idx, weight in zip(indices, weights):
                sample += weight * X_positive[idx]
            
            # Добавляем направленный шум для смещения от положительного класса
            directed_noise = np.random.normal(0, std * 0.8)
            sample = sample + directed_noise
            sample = np.maximum(sample, mean * 0.02)
            negatives.append(sample)
        
        result = np.array(negatives)
        
        # Проверка качества негативных примеров
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(result, X_positive)
        min_distances = np.min(distances, axis=1)
        
        print(f"\n📊 СТАТИСТИКА НЕГАТИВНЫХ ПРИМЕРОВ:")
        print(f"  Создано: {len(result)} образцов")
        print(f"  Мин. расстояние до положительных: {np.min(min_distances):.3f}")
        print(f"  Среднее расстояние: {np.mean(min_distances):.3f}")
        print(f"  Макс. расстояние: {np.max(min_distances):.3f}")
        
        return result
    
    def _save_training_results(self, training_summary: Dict):
        """Сохранение результатов обучения с исправленной сериализацией"""
        
        # Сохранение модели
        if self.best_model is not None:
            model_path = os.path.join(MODELS_DIR, f"user_{self.user_id}_enhanced_knn.pkl")
            
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'best_params': self.best_params,
                'validation_results': self.validation_results,
                'training_summary': training_summary
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Модель сохранена: {model_path}")
        
        # ✅ ИСПРАВЛЕНИЕ: Сохранение JSON с обработкой numpy типов
        report_path = os.path.join(DATA_DIR, f"training_report_user_{self.user_id}.json")
        
        try:
            # Создаем копию для сериализации, конвертируя numpy типы
            serializable_summary = self._make_json_serializable(training_summary)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Отчет сохранен: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Ошибка сохранения JSON отчета: {e}")
            # Сохраняем упрощенную версию
            simple_summary = {
                'user_id': self.user_id,
                'training_duration': training_summary.get('training_duration', 0),
                'dataset_size': training_summary.get('dataset_size', 0),
                'final_accuracy': training_summary.get('validation_results', {}).get('final_evaluation', {}).get('test_accuracy', 0),
                'best_params': self.best_params,
                'timestamp': training_summary.get('training_end', datetime.now().isoformat())
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(simple_summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Упрощенный отчет сохранен: {report_path}")
    
    def _make_json_serializable(self, obj):
        """Рекурсивно конвертирует объект в JSON-совместимый формат"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return str(obj)  # Комплексные числа как строки
        else:
            return obj
    
    @classmethod
    def load_trained_model(cls, user_id: int):
        """Загрузка обученной модели"""
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_enhanced_knn.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            trainer = cls(user_id)
            trainer.best_model = model_data['model']
            trainer.scaler = model_data['scaler']
            trainer.best_params = model_data['best_params']
            trainer.validation_results = model_data['validation_results']
            
            return trainer
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None
    
    def predict_with_confidence(self, features: np.ndarray) -> Tuple[bool, float, Dict]:
        """Предсказание с детальной статистикой"""
        if self.best_model is None:
            return False, 0.0, {}
        
        # Нормализация признаков
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Предсказание
        prediction = self.best_model.predict(features_scaled)[0]
        probabilities = self.best_model.predict_proba(features_scaled)[0]
        
        # Уверенность
        confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Дополнительная статистика
        distances, indices = self.best_model.kneighbors(features_scaled)
        
        detailed_stats = {
            'prediction': bool(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'nearest_distances': distances[0].tolist(),
            'model_params': self.best_params,
            'validation_accuracy': self.validation_results.get('final_evaluation', {}).get('test_accuracy', 0.0)
        }
        
        return bool(prediction), float(confidence), detailed_stats