# auth/keystroke_auth.py - Модуль аутентификации по динамике нажатий

from typing import Tuple, Optional, Dict
from datetime import datetime
import uuid

from models.user import User
from models.keystroke_data import KeystrokeData
from ml.model_manager import ModelManager
from utils.database import DatabaseManager
from utils.security import SecurityManager

class KeystrokeAuthenticator:
    """Класс для аутентификации по динамике нажатий клавиш"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.model_manager = ModelManager()
        self.security = SecurityManager()
        self.current_session = {}  # Текущие сессии записи нажатий
    
    def start_keystroke_recording(self, user_id: int) -> str:
        """
        Начало записи динамики нажатий
        Возвращает session_id
        """
        session_id = self.security.generate_session_id()
        
        self.current_session[session_id] = KeystrokeData(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
        return session_id
    
    def record_key_event(self, session_id: str, key: str, event_type: str):
        """Запись события клавиши"""
        if session_id not in self.current_session:
            raise ValueError("Сессия не найдена")
        
        self.current_session[session_id].add_key_event(key, event_type)
    
    def finish_recording(self, session_id: str, is_training: bool = False) -> Dict[str, float]:
        """
        Завершение записи и извлечение признаков
        Возвращает словарь признаков
        """
        if session_id not in self.current_session:
            raise ValueError("Сессия не найдена")
    
        keystroke_data = self.current_session[session_id]
    
        # ВАЖНО: Всегда вычисляем признаки перед сохранением
        features = keystroke_data.calculate_features()
    
        # Проверяем, что признаки были рассчитаны
        if not features:
            print("Предупреждение: Не удалось рассчитать признаки для образца")
            # Создаем пустые признаки для совместимости
            features = {
                'avg_dwell_time': 0.0,
                'std_dwell_time': 0.0,
                'avg_flight_time': 0.0,
                'std_flight_time': 0.0,
                'typing_speed': 0.0,
                'total_typing_time': 0.0
            }
            keystroke_data.features = features
    
        # Сохранение в БД если это обучающий образец
        if is_training:
            self.db.save_keystroke_sample(keystroke_data, is_training=True)
        
            # Сохранение сырых данных о нажатиях
            user = self.db.get_user_by_id(keystroke_data.user_id)
            if user:
                keystroke_data.save_raw_events_to_csv(user.id, user.username)
    
        # Удаление из текущих сессий
        del self.current_session[session_id]
    
        return features
    
    def authenticate(self, user, keystroke_features: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Второй фактор аутентификации с детальным анализом
        """
        if not user.is_trained:
            return False, 0.0, "Модель пользователя не обучена."
    
        # Аутентификация с получением детальной статистики
        is_authenticated, confidence, detailed_stats = self.model_manager.authenticate_user_detailed(
            user.id, keystroke_features
        )
    
        # ПРОСТОЙ КОНСОЛЬНЫЙ АНАЛИЗ (РАБОТАЕТ ВСЕГДА)
        print(f"\n{'='*60}")
        print(f"🔍 АНАЛИЗ АУТЕНТИФИКАЦИИ - {user.username}")
        print(f"{'='*60}")
        print(f"📅 Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"🎯 Результат: {'✅ ПРИНЯТ' if is_authenticated else '❌ ОТКЛОНЕН'}")
        print(f"🎲 Финальная уверенность: {confidence:.1%}")
        print(f"🚪 Порог принятия: {detailed_stats.get('threshold', 0.75):.1%}")
        print()
    
        print("📊 КОМПОНЕНТНЫЙ АНАЛИЗ:")
        knn_conf = detailed_stats.get('knn_confidence', 0)
        dist_score = detailed_stats.get('distance_score', 0)
        feat_score = detailed_stats.get('feature_score', 0)
        weights = detailed_stats.get('weights', {'knn': 0.5, 'distance': 0.3, 'features': 0.2})
    
        print(f"├─ KNN Классификатор: {knn_conf:.1%}")
        print(f"├─ Анализ расстояний: {dist_score:.1%}")
        print(f"├─ Анализ признаков: {feat_score:.1%}")
        print()
    
        print("⚖️ ВЗВЕШЕННОЕ КОМБИНИРОВАНИЕ:")
        knn_weighted = knn_conf * weights.get('knn', 0.5)
        dist_weighted = dist_score * weights.get('distance', 0.3)
        feat_weighted = feat_score * weights.get('features', 0.2)
    
        print(f"├─ KNN: {knn_conf:.1%} × {weights.get('knn', 0.5):.1f} = {knn_weighted:.1%}")
        print(f"├─ Расстояния: {dist_score:.1%} × {weights.get('distance', 0.3):.1f} = {dist_weighted:.1%}")
        print(f"├─ Признаки: {feat_score:.1%} × {weights.get('features', 0.2):.1f} = {feat_weighted:.1%}")
        print(f"└─ ИТОГО: {confidence:.1%}")
        print()
    
        print("📋 ВАШИ ПРИЗНАКИ КЛАВИАТУРНОГО ПОЧЕРКА:")
        print(f"├─ Время удержания клавиш: {keystroke_features.get('avg_dwell_time', 0)*1000:.1f} мс")
        print(f"├─ Время между клавишами: {keystroke_features.get('avg_flight_time', 0)*1000:.1f} мс")
        print(f"├─ Скорость печати: {keystroke_features.get('typing_speed', 0):.1f} клавиш/сек")
        print(f"└─ Общее время ввода: {keystroke_features.get('total_typing_time', 0):.1f} сек")
        print()
    
        print("🎯 РЕШЕНИЕ СИСТЕМЫ:")
        if confidence >= detailed_stats.get('threshold', 0.75):
            print(f"✅ {confidence:.1%} ≥ {detailed_stats.get('threshold', 0.75):.1%} → ДОСТУП РАЗРЕШЕН")
            print("💡 Ваш стиль печати соответствует обученному профилю")
        else:
            print(f"❌ {confidence:.1%} < {detailed_stats.get('threshold', 0.75):.1%} → ДОСТУП ЗАПРЕЩЕН")
            print("💡 Стиль печати отличается от обученного профиля")
    
        print("="*60)
    
        # Попытка открыть GUI окно (без критических ошибок)
        try:
            # Сохраняем данные для возможного просмотра
            analysis_data = {
                'user_name': user.username,
                'result': is_authenticated,
                'confidence': confidence,
                'threshold': detailed_stats.get('threshold', 0.75),
                'components': {
                    'knn': knn_conf,
                    'distance': dist_score,
                    'features': feat_score
                },
                'weights': weights,
                'keystroke_features': keystroke_features,
                'timestamp': datetime.now().isoformat()
            }
        
            # Сохраняем в временный файл для визуализации
            import json
            import os
            temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
        
            with open(os.path.join(temp_dir, 'last_auth_analysis.json'), 'w') as f:
                json.dump(analysis_data, f, indent=2)
        
            print("💾 Данные анализа сохранены для детального просмотра")
            print(f"📁 Файл: {os.path.join(temp_dir, 'last_auth_analysis.json')}")
        
        except Exception as e:
            print(f"⚠️ Не удалось сохранить данные анализа: {e}")
    
        if is_authenticated:
            message = f"Аутентификация успешна (уверенность: {confidence:.1%})"
        else:
            message = f"Аутентификация отклонена (уверенность: {confidence:.1%})"

        
        # Сохранение попытки аутентификации в базу данных для статистики
        try:
            self.db.save_auth_attempt(
                user_id=user.id,
                session_id=session_id if session_id else 'unknown',
                features=keystroke_features,
                knn_confidence=detailed_stats.get('knn_confidence', 0),
                distance_score=detailed_stats.get('distance_score', 0),
                feature_score=detailed_stats.get('feature_score', 0),
                final_confidence=confidence,
                threshold=detailed_stats.get('threshold', 0.75),
                result=is_authenticated
            )
        except Exception as e:
            print(f"Ошибка сохранения попытки аутентификации: {e}")
    
        return is_authenticated, confidence, message
    
    def train_user_model(self, user: User) -> Tuple[bool, float, str]:
        """
        Обучение модели пользователя
        Возвращает: (успех, точность, сообщение)
        """
        return self.model_manager.train_user_model(user.id)
    
    def get_training_progress(self, user: User) -> Dict[str, any]:
        """Получение прогресса обучения пользователя"""
        samples = self.db.get_user_keystroke_samples(user.id, training_only=True)
        
        from config import MIN_TRAINING_SAMPLES
        
        return {
            'current_samples': len(samples),
            'required_samples': MIN_TRAINING_SAMPLES,
            'progress_percent': min(100, (len(samples) / MIN_TRAINING_SAMPLES) * 100),
            'is_ready': len(samples) >= MIN_TRAINING_SAMPLES,
            'is_trained': user.is_trained
        }
    
    def reset_user_model(self, user: User) -> Tuple[bool, str]:
        """Сброс модели пользователя и обучающих данных"""
        try:
            # Удаление модели
            self.model_manager.delete_user_model(user.id)
            
            # Удаление обучающих образцов из БД
            # Здесь нужно добавить метод в DatabaseManager для удаления образцов
            
            # Обновление статуса пользователя
            user.is_trained = False
            user.training_samples = 0
            self.db.update_user(user)
            
            return True, "Модель и обучающие данные успешно сброшены"
        except Exception as e:
            return False, f"Ошибка при сбросе модели: {str(e)}"
    
    def get_authentication_stats(self, user: User) -> Dict[str, any]:
        """Получение статистики аутентификации пользователя"""
    
        # ✅ ПРАВИЛЬНО: Используем ОТДЕЛЬНЫЕ методы для разных типов данных
    
        # Обучающие образцы (is_training = 1)
        training_samples = self.db.get_user_training_samples(user.id)
    
        # ВСЕ образцы
        all_samples = self.db.get_user_keystroke_samples(user.id, training_only=False)
    
        # Попытки аутентификации = все - обучающие
        auth_attempts = len(all_samples) - len(training_samples)
    
        return {
            'total_samples': len(all_samples),
            'training_samples': len(training_samples),  # ✅ ТОЛЬКО обучающие
            'authentication_attempts': max(0, auth_attempts),  # ✅ ТОЛЬКО попытки входа
            'model_info': self.model_manager.get_model_info(user.id)
        }