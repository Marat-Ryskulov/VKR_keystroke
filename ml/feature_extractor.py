# ml/feature_extractor.py - Извлечение признаков из динамики нажатий

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class FeatureExtractor:
    """Класс для извлечения признаков из динамики нажатий"""
    
    @staticmethod
    def extract_features_from_samples(samples: List[dict]) -> np.ndarray:
        """Извлечение матрицы признаков из списка образцов"""
        if not samples:
            return np.array([])
        
        feature_vectors = []
        for sample in samples:
            features = sample.get('features', {})
            vector = [
                features.get('avg_dwell_time', 0),
                features.get('std_dwell_time', 0),
                features.get('avg_flight_time', 0),
                features.get('std_flight_time', 0),
                features.get('typing_speed', 0),
                features.get('total_typing_time', 0)
            ]
            feature_vectors.append(vector)
        
        return np.array(feature_vectors)
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """Нормализация признаков (z-score нормализация)"""
        if features.size == 0:
            return features, {}
        
        # Вычисление статистик для каждого признака
        stats = {}
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[1]):
            column = features[:, i]
            mean = np.mean(column)
            std = np.std(column)
            
            # Избегаем деления на ноль
            if std == 0:
                std = 1
            
            normalized[:, i] = (column - mean) / std
            stats[f'feature_{i}'] = (mean, std)
        
        return normalized, stats
    
    @staticmethod
    def apply_normalization(features: np.ndarray, stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Применение сохраненной нормализации к новым данным"""
        if features.size == 0:
            return features
        
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[1]):
            mean, std = stats.get(f'feature_{i}', (0, 1))
            normalized[:, i] = (features[:, i] - mean) / std
        
        return normalized
    
    @staticmethod
    def calculate_typing_rhythm(key_events: List[Dict]) -> Dict[str, float]:
        """Вычисление ритмических характеристик печати"""
        if len(key_events) < 3:
            return {}
        
        # Сортировка событий по времени
        sorted_events = sorted(key_events, key=lambda x: x['timestamp'])
        
        # Вычисление интервалов между нажатиями
        intervals = []
        for i in range(1, len(sorted_events)):
            if sorted_events[i]['event_type'] == 'press' and sorted_events[i-1]['event_type'] == 'press':
                interval = sorted_events[i]['timestamp'] - sorted_events[i-1]['timestamp']
                intervals.append(interval)
        
        if not intervals:
            return {}
        
        # Вычисление ритмических характеристик
        rhythm_features = {
            'rhythm_mean': np.mean(intervals),
            'rhythm_std': np.std(intervals),
            'rhythm_variation': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0,
            'rhythm_min': np.min(intervals),
            'rhythm_max': np.max(intervals)
        }
        
        return rhythm_features
    
    @staticmethod
    def extract_digraph_features(key_events: List[Dict]) -> Dict[str, float]:
        """Извлечение признаков для пар клавиш (диграфов)"""
        digraph_times = defaultdict(list)
        
        # Группировка событий по клавишам
        press_events = [e for e in key_events if e['event_type'] == 'press']
        
        # Вычисление времени между последовательными нажатиями
        for i in range(1, len(press_events)):
            prev_key = press_events[i-1]['key']
            curr_key = press_events[i]['key']
            digraph = f"{prev_key}-{curr_key}"
            
            time_diff = press_events[i]['timestamp'] - press_events[i-1]['timestamp']
            digraph_times[digraph].append(time_diff)
        
        # Агрегирование статистик по диграфам
        features = {}
        for digraph, times in digraph_times.items():
            features[f'digraph_{digraph}_mean'] = np.mean(times)
            features[f'digraph_{digraph}_std'] = np.std(times) if len(times) > 1 else 0
        
        return features