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
        self.n_neighbors = min(n_neighbors, 3)  # Максимум 3 соседа
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric='euclidean',
            weights='distance',     # Ближайшие соседи важнее
            algorithm='ball_tree'   # Более точный алгоритм
        )
        self.is_trained = False
        self.normalization_stats = None
        self.training_data = None
        
    def train(self, X_positive: np.ndarray, X_negative: np.ndarray = None) -> Tuple[bool, float]:
        """Обучение с экстремальным балансом классов"""
        n_samples = len(X_positive)
        if n_samples < 5:
            return False, 0.0
    
        print(f"\n🎯 СТРОГОЕ ОБУЧЕНИЕ")
        print(f"Твоих образцов: {n_samples}")
    
        self.training_data = X_positive.copy()
    
        # Генерируем негативные примеры
        if X_negative is None or len(X_negative) == 0:
            X_negative = self._generate_synthetic_negatives(X_positive)
    
        # КРИТИЧЕСКИ ВАЖНО: берем НАМНОГО меньше негативных
        # Соотношение 3:1 в пользу твоих данных
        neg_count = max(n_samples // 4, 5)  # Минимум 5, но обычно в 4 раза меньше
    
        if len(X_negative) > neg_count:
            # Берем САМЫЕ ДАЛЕКИЕ негативные примеры
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X_negative, X_positive)
            min_distances = np.min(distances, axis=1)
            farthest_indices = np.argsort(min_distances)[-neg_count:]
            X_negative = X_negative[farthest_indices]
    
        print(f"Финальные данные: {len(X_positive)} ТВОИХ vs {len(X_negative)} ЧУЖИХ")
        print(f"Соотношение: {len(X_positive)/(len(X_positive)+len(X_negative))*100:.0f}% твоих данных")
        print(f"Минимальное расстояние негативных: {np.min(euclidean_distances(X_negative, X_positive)):.2f}")
    
        # Обучение
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(len(X_positive)), np.zeros(len(X_negative))])
    
        self.model.fit(X, y)
        self.is_trained = True
    
        # Проверяем качество
        train_accuracy = self.model.score(X, y)
        print(f"Train accuracy: {train_accuracy:.3f}")

        self.test_data = {
            'X_positive': X_positive.copy(),
            'X_negative': X_negative.copy() if X_negative is not None else None,
            'y_positive': np.ones(len(X_positive)),
            'y_negative': np.zeros(len(X_negative)) if X_negative is not None else None
        }
    
        return True, train_accuracy
    
    def authenticate(self, features: np.ndarray, threshold: float = 0.5, verbose: bool = False) -> Tuple[bool, float, dict]:
        """
        Продвинутая аутентификация с детальной статистикой
        Возвращает: (результат, финальная_уверенность, детальная_статистика)
        """
        if not self.is_trained:
            return False, 0.0, {}

        # Убеждаемся, что features - это 1D массив
        if features.ndim > 1:
            features = features.flatten()

        if verbose:
            print(f"\n=== НАЧАЛО АУТЕНТИФИКАЦИИ ===")
            print(f"Входящие признаки: {features}")

        # 1. Основное предсказание KNN
        features_reshaped = features.reshape(1, -1)
        probabilities = self.model.predict_proba(features_reshaped)[0]

        # Получаем базовую вероятность
        knn_probability = 0.0
        if len(probabilities) > 1 and 1.0 in self.model.classes_:
            class_1_index = list(self.model.classes_).index(1.0)
            knn_probability = probabilities[class_1_index]

        # 2. Анализ расстояний
        distance_score = 0.0
        distance_details = {}

        if hasattr(self, 'training_data') and self.training_data is not None:
            from sklearn.metrics.pairwise import euclidean_distances
    
            X_positive = self.training_data
            distances = euclidean_distances(features_reshaped, X_positive)[0]
    
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
    
            # Статистика обучающих данных
            if len(X_positive) > 1:
                train_distances = euclidean_distances(X_positive, X_positive)
                train_distances = train_distances[train_distances > 0]
                mean_train_distance = np.mean(train_distances)
                std_train_distance = np.std(train_distances)
            else:
                mean_train_distance = 1.0
                std_train_distance = 0.5
    
            # Оценки расстояний
            norm_min = min_distance / (mean_train_distance + 1e-6)
            distance_score = max(0, 1.0 - norm_min * 0.8)
    
            distance_details = {
                'min_distance': min_distance,
                'mean_distance': mean_distance,
                'mean_train_distance': mean_train_distance,
                'normalized_distance': norm_min
            }

        # 3. Анализ признаков
        feature_score = 1.0
        feature_details = {}

        if hasattr(self, 'training_data') and self.training_data is not None:
            X_positive = self.training_data
            train_mean = np.mean(X_positive, axis=0)
            train_std = np.std(X_positive, axis=0)
    
            feature_penalties = []
            feature_names = ['avg_dwell', 'std_dwell', 'avg_flight', 'std_flight', 'speed', 'total_time']
    
            for i, (feat_val, train_m, train_s, name) in enumerate(zip(features, train_mean, train_std, feature_names)):
                if train_s > 0:
                    z_score_val = abs(feat_val - train_m) / train_s
                    if hasattr(z_score_val, '__len__'):
                        z_score_val = float(z_score_val)
                    penalty = min(0.3, z_score_val * 0.1)
                else:
                    penalty = 0.0
                    z_score_val = 0.0
        
                feature_penalties.append(penalty)
                feature_details[name] = {
                    'value': float(feat_val),
                    'train_mean': float(train_m),
                    'train_std': float(train_s),
                    'z_score': float(z_score_val),
                    'penalty': penalty
                }
    
            total_penalty = sum(feature_penalties) / len(feature_penalties)
            feature_score = max(0.1, 1.0 - total_penalty)

        # 4. Комбинирование оценок
        if len(self.training_data) >= 30:
            weights = {'knn': 0.5, 'distance': 0.3, 'features': 0.2}
        elif len(self.training_data) >= 15:
            weights = {'knn': 0.4, 'distance': 0.4, 'features': 0.2}
        else:
            weights = {'knn': 0.2, 'distance': 0.6, 'features': 0.2}

        final_probability = (
            weights['knn'] * knn_probability +
            weights['distance'] * distance_score +
            weights['features'] * feature_score
        )

        # Принятие решения
        is_authenticated = final_probability >= threshold

        # Детальная статистика
        detailed_stats = {
            'knn_confidence': knn_probability,
            'distance_score': distance_score,
            'feature_score': feature_score,
            'final_confidence': final_probability,
            'threshold': threshold,
            'weights': weights,
            'distance_details': distance_details,
            'feature_details': feature_details,
            'training_samples': len(self.training_data) if hasattr(self, 'training_data') else 0
        }

        if verbose:
            print(f"\nФИНАЛЬНЫЕ ОЦЕНКИ:")
            print(f"  KNN: {knn_probability:.3f} (вес: {weights['knn']})")
            print(f"  Distance: {distance_score:.3f} (вес: {weights['distance']})")
            print(f"  Features: {feature_score:.3f} (вес: {weights['features']})")
            print(f"  Final: {final_probability:.3f} (порог: {threshold})")
            print(f"  РЕЗУЛЬТАТ: {'ПРИНЯТ' if is_authenticated else 'ОТКЛОНЕН'}")
            print(f"=== КОНЕЦ АУТЕНТИФИКАЦИИ ===\n")

        return is_authenticated, final_probability, detailed_stats
    
    def _generate_synthetic_negatives(self, X_positive: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Генерация РЕАЛИСТИЧНЫХ негативных примеров для обучения"""
        n_samples = len(X_positive)
        n_features = X_positive.shape[1]

        # Анализируем ТВОИ данные
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
        min_vals = np.min(X_positive, axis=0)
        max_vals = np.max(X_positive, axis=0)

        print(f"\n🔍 АНАЛИЗ ТВОИХ ДАННЫХ:")
        print(f"  Удержание клавиш: {mean[0]*1000:.1f} ± {std[0]*1000:.1f} мс")
        print(f"  Время между клавишами: {mean[2]*1000:.1f} ± {std[2]*1000:.1f} мс")
        print(f"  Скорость печати: {mean[4]:.1f} ± {std[4]:.1f} кл/с")
        print(f"  Общее время: {mean[5]:.1f} ± {std[5]:.1f} сек")

        synthetic_samples = []

        # Стратегия 1: Близкие конкуренты (25%) - труднее всего отличить
        close_count = n_samples // 4
        print(f"Создаем {close_count} БЛИЗКИХ КОНКУРЕНТОВ...")
        for i in range(close_count):
            # Небольшие, но систематические отличия в 1-2 признаках
            sample = mean.copy()
        
            # Выбираем 1-2 признака для изменения
            features_to_change = np.random.choice(6, size=np.random.randint(1, 3), replace=False)
        
            for feat_idx in features_to_change:
                if feat_idx in [0, 1]:  # время удержания
                    factor = np.random.choice([0.7, 1.4])  # быстрее или медленнее удержание
                elif feat_idx in [2, 3]:  # время между клавишами
                    factor = np.random.choice([0.6, 1.6])  # быстрее или медленнее переходы
                elif feat_idx == 4:  # скорость
                    factor = np.random.choice([0.8, 1.3])  # немного другая скорость
                else:  # общее время
                    factor = np.random.choice([0.75, 1.35])
            
                sample[feat_idx] = mean[feat_idx] * factor
        
            # Добавляем небольшой шум
            noise = np.random.normal(0, std * 0.3)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.1)  # не даем стать слишком маленькими
            synthetic_samples.append(sample)

        # Стратегия 2: Другой стиль печати (40%)
        different_style_count = int(n_samples * 0.4)
        print(f"Создаем {different_style_count} пользователей с ДРУГИМ СТИЛЕМ...")
        for i in range(different_style_count):
            # Более заметные, но реалистичные отличия
            if np.random.random() < 0.5:
                # "Охотники на клавиатуре" - быстрые и точные
                style_factors = np.array([
                    np.random.uniform(0.4, 0.8),     # быстрое удержание
                    np.random.uniform(0.5, 0.9),     # стабильное удержание
                    np.random.uniform(0.3, 0.7),     # быстрые переходы
                    np.random.uniform(0.4, 0.8),     # стабильные переходы
                    np.random.uniform(1.2, 2.5),     # высокая скорость
                    np.random.uniform(0.4, 0.8)      # меньше времени
                ])
            else:
                # "Размышляющие печатающие" - медленные и вдумчивые
                style_factors = np.array([
                    np.random.uniform(1.3, 2.5),     # долгое удержание
                    np.random.uniform(1.1, 2.0),     # вариативное удержание
                    np.random.uniform(1.5, 3.5),     # долгие паузы
                    np.random.uniform(1.2, 2.8),     # вариативные паузы
                    np.random.uniform(0.3, 0.8),     # низкая скорость
                    np.random.uniform(1.3, 2.5)      # больше времени
                ])
        
            sample = mean * style_factors
            # Добавляем индивидуальный шум
            noise = np.random.normal(0, std * 0.5)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.05)
            synthetic_samples.append(sample)

        # Стратегия 3: Адаптивные имитаторы (25%) - пытаются подражать, но не идеально
        adaptive_count = int(n_samples * 0.25)
        print(f"Создаем {adaptive_count} АДАПТИВНЫХ ИМИТАТОРОВ...")
        for i in range(adaptive_count):
            # Берем ваши средние значения как основу, но добавляем систематические ошибки
            sample = mean.copy()
        
            # Имитатор не может точно воспроизвести все характеристики
            # Он близок по 3-4 признакам, но ошибается в 2-3
            success_rate = np.random.uniform(0.6, 0.8)  # успешность имитации
        
            for j in range(6):
                if np.random.random() > success_rate:
                    # Ошибка в имитации этого признака
                    if j in [0, 2]:  # времена - сложно имитировать точно
                        error_factor = np.random.uniform(0.6, 1.7)
                    else:  # остальные признаки
                        error_factor = np.random.uniform(0.7, 1.5)
                    sample[j] = mean[j] * error_factor
                else:
                    # Успешная имитация - близко к вашим значениям
                    sample[j] = mean[j] * np.random.uniform(0.9, 1.1)
        
            # Имитаторы обычно менее стабильны
            instability_noise = np.random.normal(0, std * 0.8)
            sample = sample + instability_noise
            sample = np.maximum(sample, mean * 0.1)
            synthetic_samples.append(sample)

        # Стратегия 4: Экстремальные случаи (10%) - сильно отличающиеся
        extreme_count = n_samples - close_count - different_style_count - adaptive_count
        print(f"Создаем {extreme_count} ЭКСТРЕМАЛЬНЫХ пользователей...")
        for i in range(extreme_count):
            if np.random.random() < 0.5:
                # Супер-быстрые
                extreme_factors = np.array([
                    np.random.uniform(0.1, 0.5),     # очень быстрое удержание
                    np.random.uniform(0.2, 0.6),     # низкая вариативность
                    np.random.uniform(0.05, 0.3),    # очень быстрые переходы
                    np.random.uniform(0.1, 0.5),     # стабильные переходы
                    np.random.uniform(3.0, 8.0),     # очень высокая скорость
                    np.random.uniform(0.1, 0.4)      # очень мало времени
                ])
            else:
                # Супер-медленные
                extreme_factors = np.array([
                    np.random.uniform(3.0, 8.0),     # очень долгое удержание
                    np.random.uniform(2.5, 6.0),     # очень высокая вариативность
                    np.random.uniform(4.0, 12.0),    # очень долгие паузы
                    np.random.uniform(3.0, 8.0),     # очень вариативные паузы
                    np.random.uniform(0.1, 0.4),     # очень низкая скорость
                    np.random.uniform(3.0, 8.0)      # очень много времени
                ])
        
            sample = mean * extreme_factors
            noise = np.random.normal(0, std * 0.3)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.01)
            synthetic_samples.append(sample)

        result = np.array(synthetic_samples)

        # Проверяем качество негативных примеров
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(result, X_positive)
        min_distances = np.min(distances, axis=1)

        print(f"\n📊 СТАТИСТИКА РЕАЛИСТИЧНЫХ НЕГАТИВНЫХ ПРИМЕРОВ:")
        print(f"  Создано: {len(result)} образцов")
        print(f"  Минимальное расстояние до твоих данных: {np.min(min_distances):.3f}")
        print(f"  Среднее расстояние: {np.mean(min_distances):.3f}")
        print(f"  Максимальное расстояние: {np.max(min_distances):.3f}")
    
        # Категоризация по сложности
        very_close = np.sum(min_distances < np.std(min_distances) * 0.5)
        close = np.sum((min_distances >= np.std(min_distances) * 0.5) & 
                    (min_distances < np.mean(min_distances)))
        far = len(min_distances) - very_close - close
    
        print(f"  Очень близкие (трудные): {very_close}")
        print(f"  Умеренно близкие: {close}")
        print(f"  Далекие (легкие): {far}")

        # Убираем слишком близкие примеры (которые могут быть ошибочно приняты)
        distance_threshold = np.percentile(min_distances, 20)  # убираем 20% самых близких
        good_indices = min_distances >= distance_threshold
        result_filtered = result[good_indices]

        print(f"  После фильтрации слишком близких: {len(result_filtered)} образцов")
    
        if len(result_filtered) < len(result) * 0.7:  # если убрали больше 30%
            print("  Предупреждение: убрано много близких примеров - возможно, нужно больше разнообразия")
    
        return result_filtered
    
    def save_model(self, user_id: int):
        """Сохранение модели на диск"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
    
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
    
        model_data = {
            'model': self.model,
            'n_neighbors': self.n_neighbors,
            'normalization_stats': self.normalization_stats,
            'is_trained': self.is_trained,
            'training_data': self.training_data,  # Сохраняем обучающие данные
            'test_data': self.test_data  # ✅ Сохраняем тестовые данные
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
            authenticator.training_data = model_data.get('training_data', None)  # Совместимость с старыми моделями

            authenticator.test_data = model_data.get('test_data', None)  # ✅ Загружаем тестовые данные
        
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