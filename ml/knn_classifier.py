# ml/knn_classifier.py - K-ближайших соседей классификатор

import numpy as np
from typing import Tuple, Optional, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle

from config import KNN_NEIGHBORS, MODELS_DIR
import os

class KNNAuthenticator:
    """KNN классификатор для аутентификации по динамике нажатий"""
    
    def __init__(self, n_neighbors: int = KNN_NEIGHBORS):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='euclidean',
            weights='distance'  # Веса обратно пропорциональны расстоянию
        )
        self.is_trained = False
        self.normalization_stats = None
        
    def train(self, X_positive: np.ndarray, X_negative: np.ndarray = None) -> Tuple[bool, float]:
        """
        Обучение модели
        X_positive: образцы легитимного пользователя
        X_negative: образцы других пользователей (опционально)
        """
        if len(X_positive) < self.n_neighbors:
            return False, 0.0
        
        # Подготовка данных
        if X_negative is None or len(X_negative) == 0:
            # Генерация синтетических негативных примеров
            X_negative = self._generate_synthetic_negatives(X_positive)
        
        # Объединение данных
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([
            np.ones(len(X_positive)),     # 1 для легитимного пользователя
            np.zeros(len(X_negative))     # 0 для остальных
        ])
        
        # Обучение модели
        self.model.fit(X, y)
        self.is_trained = True
        
        # Оценка точности с кросс-валидацией
        if len(X) >= 5:  # Минимум для 5-fold CV
            scores = cross_val_score(self.model, X, y, cv=min(5, len(X)))
            accuracy = scores.mean()
        else:
            # Простая оценка на обучающих данных
            accuracy = self.model.score(X, y)
        
        return True, accuracy
    
    def authenticate(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Аутентификация пользователя
        Возвращает (успех, уверенность)
        """
        if not self.is_trained:
            return False, 0.0
        
        # Предсказание вероятностей
        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        
        # Вероятность того, что это легитимный пользователь
        auth_probability = probabilities[1]
        
        # Принятие решения
        is_authenticated = auth_probability >= threshold
        
        return is_authenticated, auth_probability
    
    def _generate_synthetic_negatives(self, X_positive: np.ndarray, factor: float = 2.0) -> np.ndarray:
        """Генерация синтетических негативных примеров"""
        n_samples = len(X_positive)
        n_features = X_positive.shape[1]
        
        # Вычисление статистик положительных примеров
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
        
        # Генерация негативных примеров с большим разбросом
        synthetic = np.random.normal(
            loc=mean,
            scale=std * factor,
            size=(n_samples * 2, n_features)
        )
        
        # Добавление выбросов
        outliers = np.random.uniform(
            low=mean - 3 * std,
            high=mean + 3 * std,
            size=(n_samples, n_features)
        )
        
        return np.vstack([synthetic, outliers])
    
    def save_model(self, user_id: int):
        """Сохранение модели на диск"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
        
        model_data = {
            'model': self.model,
            'n_neighbors': self.n_neighbors,
            'normalization_stats': self.normalization_stats,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, user_id: int) -> Optional['KNNAuthenticator']:
        """Загрузка модели с диска"""
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            authenticator = cls(n_neighbors=model_data['n_neighbors'])
            authenticator.model = model_data['model']
            authenticator.normalization_stats = model_data['normalization_stats']
            authenticator.is_trained = model_data['is_trained']
            
            return authenticator
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Получение важности признаков (для анализа)"""
        if not self.is_trained:
            return []
        
        # Для KNN можно оценить важность через дисперсию
        feature_names = [
            'avg_dwell_time',
            'std_dwell_time', 
            'avg_flight_time',
            'std_flight_time',
            'typing_speed',
            'total_typing_time'
        ]
        
        # Получаем обучающие данные
        X_train = self.model._fit_X
        
        # Вычисляем дисперсию для каждого признака
        variances = np.var(X_train, axis=0)
        
        # Нормализуем важности
        importance = variances / np.sum(variances)
        
        return list(zip(feature_names, importance))