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
    
    def authenticate(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Продвинутая аутентификация с множественными методами
        """
        if not self.is_trained:
            return False, 0.0
    
        print(f"\n=== НАЧАЛО АУТЕНТИФИКАЦИИ ===")
        print(f"Входящие признаки: {features}")
    
        # 1. Основное предсказание KNN
        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        print(f"KNN probabilities: {probabilities}")
        print(f"KNN classes: {self.model.classes_}")
    
        # Получаем базовую вероятность
        knn_probability = 0.0
        if len(probabilities) > 1 and 1.0 in self.model.classes_:
            class_1_index = list(self.model.classes_).index(1.0)
            knn_probability = probabilities[class_1_index]
    
        print(f"KNN auth probability: {knn_probability:.3f}")
    
        # 2. Анализ расстояний до обучающих образцов
        distance_score = 0.0
        if hasattr(self, 'training_data') and self.training_data is not None:
            from sklearn.metrics.pairwise import euclidean_distances
        
            X_positive = self.training_data
            distances = euclidean_distances(features.reshape(1, -1), X_positive)[0]
        
            # Детальная статистика расстояний
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
        
            # k ближайших соседей
            k = min(5, len(distances))
            nearest_distances = np.sort(distances)[:k]
            avg_nearest = np.mean(nearest_distances)
        
            # Статистика обучающих образцов
            if len(X_positive) > 1:
                train_distances = euclidean_distances(X_positive, X_positive)
                train_distances = train_distances[train_distances > 0]
                mean_train_distance = np.mean(train_distances)
                std_train_distance = np.std(train_distances)
                median_train_distance = np.median(train_distances)
            else:
                mean_train_distance = 1.0
                std_train_distance = 0.5
                median_train_distance = 1.0
        
            print(f"\nАНАЛИЗ РАССТОЯНИЙ:")
            print(f"  Min distance: {min_distance:.4f}")
            print(f"  Mean distance: {mean_distance:.4f}")
            print(f"  Median distance: {median_distance:.4f}")
            print(f"  Avg nearest {k}: {avg_nearest:.4f}")
            print(f"  Train mean: {mean_train_distance:.4f}")
            print(f"  Train std: {std_train_distance:.4f}")
            print(f"  Train median: {median_train_distance:.4f}")
        
            # Множественные оценки расстояний
            scores = []
        
            # Оценка 1: Минимальное расстояние
            norm_min = min_distance / (mean_train_distance + 1e-6)
            score1 = max(0, 1.0 - norm_min * 0.8)
            scores.append(('min_dist', score1))
        
            # Оценка 2: Среднее расстояние до ближайших
            norm_nearest = avg_nearest / (mean_train_distance + 1e-6)
            score2 = max(0, 1.0 - norm_nearest * 0.6)
            scores.append(('nearest_avg', score2))
        
            # Оценка 3: Сравнение с медианой
            norm_median = min_distance / (median_train_distance + 1e-6)
            score3 = max(0, 1.0 - norm_median * 0.7)
            scores.append(('median_comp', score3))
        
            # Оценка 4: Z-score анализ
            if std_train_distance > 0:
                z_score = abs(min_distance - mean_train_distance) / std_train_distance
                score4 = max(0, 1.0 - z_score * 0.3)
            else:
                score4 = score1
            scores.append(('z_score', score4))
        
            # Комбинированная оценка расстояний
            distance_score = np.mean([s[1] for s in scores])
        
            print(f"  Distance scores: {scores}")
            print(f"  Combined distance score: {distance_score:.3f}")
    
        # 3. Анализ признаков (проверка на разумность)
        feature_score = 1.0
        if hasattr(self, 'training_data') and self.training_data is not None:
            X_positive = self.training_data
        
            # Статистики обучающих данных
            train_mean = np.mean(X_positive, axis=0)
            train_std = np.std(X_positive, axis=0)
        
            # Проверяем каждый признак
            feature_penalties = []
            feature_names = ['avg_dwell', 'std_dwell', 'avg_flight', 'std_flight', 'speed', 'total_time']
        
            for i, (feat_val, train_m, train_s, name) in enumerate(zip(features, train_mean, train_std, feature_names)):
                if train_s > 0:
                    z_score = abs(feat_val - train_m) / train_s
                    penalty = min(0.3, z_score * 0.1)  # Максимальный штраф 30%
                else:
                    penalty = 0.0
            
                feature_penalties.append(penalty)
                print(f"  {name}: val={feat_val:.4f}, train_mean={train_m:.4f}, z_score={(abs(feat_val - train_m) / (train_s + 1e-6)):.2f}, penalty={penalty:.3f}")
        
            # Применяем штрафы
            total_penalty = sum(feature_penalties) / len(feature_penalties)
            feature_score = max(0.1, 1.0 - total_penalty)
        
            print(f"  Feature analysis score: {feature_score:.3f}")
    
        # 4. Адаптивное комбинирование оценок
        if len(self.training_data) >= 30:
            # Для больших данных больше доверяем ML модели
            weights = {'knn': 0.5, 'distance': 0.3, 'features': 0.2}
        elif len(self.training_data) >= 15:
            # Для средних данных балансируем
            weights = {'knn': 0.4, 'distance': 0.4, 'features': 0.2}
        else:
            # Для малых данных больше доверяем расстояниям
            weights = {'knn': 0.2, 'distance': 0.6, 'features': 0.2}
    
        # Финальная оценка
        final_probability = (
            weights['knn'] * knn_probability +
            weights['distance'] * distance_score +
            weights['features'] * feature_score
        )
    
        print(f"\nФИНАЛЬНЫЕ ОЦЕНКИ:")
        print(f"  KNN: {knn_probability:.3f} (вес: {weights['knn']})")
        print(f"  Distance: {distance_score:.3f} (вес: {weights['distance']})")
        print(f"  Features: {feature_score:.3f} (вес: {weights['features']})")
        print(f"  Final probability: {final_probability:.3f}")
        print(f"  Threshold: {threshold}")
    
        # Принятие решения
        is_authenticated = final_probability >= threshold
    
        print(f"  РЕЗУЛЬТАТ: {'ПРИНЯТ' if is_authenticated else 'ОТКЛОНЕН'}")
        print(f"=== КОНЕЦ АУТЕНТИФИКАЦИИ ===\n")
    
        return is_authenticated, final_probability
    
    def _generate_synthetic_negatives(self, X_positive: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Генерация ЭКСТРЕМАЛЬНО отличающихся негативных примеров"""
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
    
        # 1. ЧЕРЕПАХИ (в 20 раз медленнее)
        print("Создаем ЧЕРЕПАХ...")
        for i in range(n_samples // 2):
            sample = np.array([
                mean[0] * np.random.uniform(15, 25),    # удержание в 15-25 раз дольше
                mean[1] * np.random.uniform(10, 20),    # вариативность удержания  
                mean[2] * np.random.uniform(20, 40),    # паузы в 20-40 раз дольше
                mean[3] * np.random.uniform(15, 30),    # вариативность пауз
                mean[4] * np.random.uniform(0.02, 0.08), # скорость в 12-50 раз медленнее
                mean[5] * np.random.uniform(15, 30)     # время в 15-30 раз больше
            ])
            synthetic_samples.append(sample)
    
        # 2. ГЕПАРДЫ (в 20 раз быстрее)  
        print("Создаем ГЕПАРДОВ...")
        for i in range(n_samples // 2):
            sample = np.array([
                mean[0] * np.random.uniform(0.02, 0.08), # удержание в 12-50 раз короче
                mean[1] * np.random.uniform(0.05, 0.15), # низкая вариативность
                mean[2] * np.random.uniform(0.01, 0.05), # паузы в 20-100 раз короче
                mean[3] * np.random.uniform(0.02, 0.10), # очень стабильно
                mean[4] * np.random.uniform(15, 40),     # скорость в 15-40 раз быстрее
                mean[5] * np.random.uniform(0.05, 0.20)  # время в 5-20 раз меньше
            ])
            synthetic_samples.append(sample)
    
        # 3. РОБОТЫ (нулевая вариативность)
        print("Создаем РОБОТОВ...")
        for i in range(n_samples // 3):
            base_dwell = np.random.uniform(0.01, 0.30)
            base_flight = np.random.uniform(0.01, 0.50)
            sample = np.array([
                base_dwell,                             # фиксированное удержание
                base_dwell * 0.001,                     # почти нет вариативности
                base_flight,                            # фиксированные паузы  
                base_flight * 0.001,                    # почти нет вариативности
                1.0 / (base_dwell + base_flight + 0.01), # математически точная скорость
                43 * (base_dwell + base_flight)         # точное время для 43 символов
            ])
            synthetic_samples.append(sample)
    
        # 4. ХАОС (огромная вариативность)
        print("Создаем ХАОС...")
        for i in range(n_samples // 3):
            # Основные значения в диапазоне твоих, но вариативность ОГРОМНАЯ
            base_dwell = mean[0] * np.random.uniform(0.5, 2.0)
            base_flight = mean[2] * np.random.uniform(0.3, 3.0)
        
            sample = np.array([
                base_dwell,                             # среднее удержание
                base_dwell * np.random.uniform(5, 15),  # ОГРОМНАЯ вариативность удержания
                base_flight,                            # среднее между клавишами
                base_flight * np.random.uniform(8, 20), # ОГРОМНАЯ вариативность пауз
                mean[4] * np.random.uniform(0.3, 3.0),  # непредсказуемая скорость
                mean[5] * np.random.uniform(0.5, 4.0)   # непредсказуемое время
            ])
            synthetic_samples.append(sample)
    
        # 5. ИНОПЛАНЕТЯНЕ (из другой вселенной)
        print("Создаем ИНОПЛАНЕТЯН...")
        for i in range(n_samples // 3):
            sample = np.array([
                np.random.uniform(0.005, 3.0),          # любое удержание
                np.random.uniform(0.001, 2.0),          # любая вариативность
                np.random.uniform(0.005, 8.0),          # любые паузы
                np.random.uniform(0.001, 4.0),          # любая вариативность
                np.random.uniform(0.1, 100.0),          # любая скорость
                np.random.uniform(1.0, 200.0)           # любое время
            ])
            synthetic_samples.append(sample)
    
        result = np.array(synthetic_samples)
    
        # Проверяем расстояния
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(result, X_positive)
        min_distances = np.min(distances, axis=1)
    
        print(f"\n📊 СТАТИСТИКА НЕГАТИВНЫХ ПРИМЕРОВ:")
        print(f"  Создано: {len(result)} образцов")
        print(f"  Минимальное расстояние до твоих данных: {np.min(min_distances):.2f}")
        print(f"  Среднее расстояние: {np.mean(min_distances):.2f}")
        print(f"  Максимальное расстояние: {np.max(min_distances):.2f}")
    
        # Убираем примеры, которые слишком близко к твоим данным
        threshold = np.mean(min_distances)  # Средний порог
        far_indices = min_distances >= threshold
        result_filtered = result[far_indices]
    
        print(f"  После фильтрации: {len(result_filtered)} образцов (убрали слишком близкие)")
        print(f"  Минимальное расстояние после фильтрации: {np.min(min_distances[far_indices]):.2f}")
    
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