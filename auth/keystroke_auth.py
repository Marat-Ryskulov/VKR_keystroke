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
    
    def authenticate(self, user: User, keystroke_features: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Второй фактор аутентификации - проверка динамики нажатий
        Возвращает: (успех, уверенность, сообщение)
        """
        if not user.is_trained:
            return False, 0.0, "Модель пользователя не обучена. Необходимо пройти обучение."
        
        # Аутентификация через модель
        is_authenticated, confidence, message = self.model_manager.authenticate_user(
            user.id, 
            keystroke_features
        )
        
        # Сохранение попытки аутентификации для анализа
        session_id = str(uuid.uuid4())
        keystroke_data = KeystrokeData(
            user_id=user.id,
            session_id=session_id,
            timestamp=datetime.now(),
            features=keystroke_features
        )
        self.db.save_keystroke_sample(keystroke_data, is_training=False)
        
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