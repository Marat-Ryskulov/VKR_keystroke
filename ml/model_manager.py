# ml/model_manager.py - Менеджер для управления моделями машинного обучения

import numpy as np
from typing import Optional, Tuple, List
import os

from ml.knn_classifier import KNNAuthenticator
from ml.feature_extractor import FeatureExtractor
from utils.database import DatabaseManager
from config import MIN_TRAINING_SAMPLES, THRESHOLD_ACCURACY, MODELS_DIR

class ModelManager:
    """Менеджер для управления моделями пользователей"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.feature_extractor = FeatureExtractor()
        self.models_cache = {}  # Кэш загруженных моделей
    
    def train_user_model(self, user_id: int) -> Tuple[bool, float, str]:
        """
        Обучение модели для пользователя
        Возвращает: (успех, точность, сообщение)
        """
        # Получение образцов пользователя
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        if len(user_samples) < MIN_TRAINING_SAMPLES:
            return False, 0.0, f"Недостаточно образцов. Необходимо минимум {MIN_TRAINING_SAMPLES}, собрано {len(user_samples)}"
        
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
        user = self.db.get_user_by_username(self._get_username_by_id(user_id))
        if user:
            user.is_trained = True
            user.training_samples = len(user_samples)
            self.db.update_user(user)
        
        # Добавление в кэш
        self.models_cache[user_id] = classifier
        
        return True, accuracy, f"Модель успешно обучена с точностью {accuracy:.2%}"
    
    def authenticate_user(self, user_id: int, keystroke_features: dict) -> Tuple[bool, float, str]:
        """
        Аутентификация пользователя по динамике нажатий
        Возвращает: (успех, уверенность, сообщение)
        """
        # Получение модели
        classifier = self._get_user_model(user_id)
        if classifier is None:
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
        
        # Нормализация признаков
        if classifier.normalization_stats:
            feature_vector_norm = self.feature_extractor.apply_normalization(
                feature_vector.reshape(1, -1), 
                classifier.normalization_stats
            ).flatten()
        else:
            feature_vector_norm = feature_vector
        
        # Аутентификация
        is_authenticated, confidence = classifier.authenticate(feature_vector_norm, THRESHOLD_ACCURACY)
        
        if is_authenticated:
            message = f"Аутентификация успешна (уверенность: {confidence:.2%})"
        else:
            message = f"Аутентификация отклонена (уверенность: {confidence:.2%})"
        
        return is_authenticated, confidence, message
    
    def _get_user_model(self, user_id: int) -> Optional[KNNAuthenticator]:
        """Получение модели пользователя (с кэшированием)"""
        # Проверка кэша
        if user_id in self.models_cache:
            return self.models_cache[user_id]
        
        # Загрузка с диска
        classifier = KNNAuthenticator.load_model(user_id)
        if classifier:
            self.models_cache[user_id] = classifier
        
        return classifier
    
    def _get_negative_samples(self, exclude_user_id: int) -> Optional[np.ndarray]:
        """Получение образцов других пользователей для негативных примеров"""
        all_negative_samples = []
        
        # Получаем всех пользователей кроме текущего
        all_users = self.db.get_all_users()
        for user in all_users:
            if user.id != exclude_user_id:
                user_samples = self.db.get_user_keystroke_samples(user.id, training_only=True)
                if user_samples:
                    features = self.feature_extractor.extract_features_from_samples(user_samples)
                    all_negative_samples.append(features)
        
        if all_negative_samples:
            return np.vstack(all_negative_samples)
        return None
    
    def _get_username_by_id(self, user_id: int) -> Optional[str]:
        """Получение имени пользователя по ID"""
        users = self.db.get_all_users()
        for user in users:
            if user.id == user_id:
                return user.username
        return None
    
    def delete_user_model(self, user_id: int):
        """Удаление модели пользователя"""
        # Удаление из кэша
        if user_id in self.models_cache:
            del self.models_cache[user_id]
        
        # Удаление файла модели
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def get_model_info(self, user_id: int) -> Optional[dict]:
        """Получение информации о модели пользователя"""
        classifier = self._get_user_model(user_id)
        if not classifier:
            return None
        
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        return {
            'is_trained': classifier.is_trained,
            'n_neighbors': classifier.n_neighbors,
            'training_samples': len(user_samples),
            'feature_importance': classifier.get_feature_importance()
        }