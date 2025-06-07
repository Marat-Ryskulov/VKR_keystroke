# ml/random_forest_classifier.py - Random Forest классификатор

import numpy as np
from typing import Tuple, Optional, List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import os

from config import MODELS_DIR


class RandomForestAuthenticator:
    """Random Forest для аутентификации по динамике нажатий"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.cv_score = 0.0
        self.n_samples = 0
        self.feature_importances = None
        self.best_params = {}
    
    def train(self, features: List[List[float]], optimize_params: bool = True) -> Tuple[float, float]:
        """
        Обучение модели Random Forest
        
        Args:
            features: Список векторов признаков
            optimize_params: Оптимизировать ли гиперпараметры
            
        Returns:
            Tuple[точность, стандартное отклонение]
        """
        if len(features) < 5:
            raise ValueError("Недостаточно образцов для обучения")
        
        # Подготовка данных
        X_train = np.array(features)
        y_train = np.ones(len(features))  # Все образцы принадлежат одному пользователю
        
        if optimize_params and len(features) >= 10:
            # Поиск оптимальных параметров
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=min(5, len(features)), 
                scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
        else:
            # Использование параметров по умолчанию
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.n_samples = len(X_train)
        
        # Сохранение важности признаков
        self.feature_importances = self.model.feature_importances_
        
        # Оценка точности через кросс-валидацию
        scores = cross_val_score(self.model, X_train, y_train, cv=min(5, len(features)))
        accuracy = scores.mean()
        self.cv_score = accuracy
        
        return accuracy, scores.std()
    
    def authenticate(self, features: List[float], threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Аутентификация пользователя
        
        Args:
            features: Вектор признаков для проверки
            threshold: Порог уверенности
            
        Returns:
            Tuple[успех, уверенность]
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Преобразование в нужный формат
        X_test = np.array([features])
        
        # Получение вероятностей
        probabilities = self.model.predict_proba(X_test)[0]
        confidence = probabilities[1]  # Вероятность принадлежности пользователю
        
        # Дополнительная проверка через proximity (близость к обучающим образцам)
        # Это делает модель более консервативной
        decision_function = self.model.decision_function(X_test)[0]
        normalized_confidence = 1 / (1 + np.exp(-decision_function))  # Сигмоида
        
        # Комбинированная уверенность
        final_confidence = (confidence + normalized_confidence) / 2
        
        return final_confidence >= threshold, final_confidence
    
    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """Получение важности признаков"""
        if not self.is_trained or self.feature_importances is None:
            return None
        
        feature_names = [
            'avg_dwell_time',
            'std_dwell_time', 
            'avg_flight_time',
            'std_flight_time',
            'typing_speed',
            'total_typing_time'
        ]
        
        return dict(zip(feature_names, self.feature_importances))
    
    def save(self, filepath: str):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'cv_score': self.cv_score,
            'n_samples': self.n_samples,
            'feature_importances': self.feature_importances,
            'best_params': self.best_params
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestAuthenticator':
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        authenticator = cls(
            n_estimators=model_data['n_estimators'],
            max_depth=model_data['max_depth'],
            random_state=model_data['random_state']
        )
        
        authenticator.model = model_data['model']
        authenticator.is_trained = model_data['is_trained']
        authenticator.cv_score = model_data.get('cv_score', 0.0)
        authenticator.n_samples = model_data.get('n_samples', 0)
        authenticator.feature_importances = model_data.get('feature_importances')
        authenticator.best_params = model_data.get('best_params', {})
        
        return authenticator