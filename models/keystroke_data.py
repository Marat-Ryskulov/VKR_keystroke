# models/keystroke_data.py - Модель данных динамики нажатий

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime
import time

@dataclass
class KeyEvent:
    """Событие нажатия/отпускания клавиши"""
    key: str
    event_type: str  # 'press' или 'release'
    timestamp: float
    
@dataclass
class KeystrokeData:
    """Данные о динамике набора текста"""
    
    user_id: int
    session_id: str
    timestamp: datetime
    key_events: List[KeyEvent] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    
    def add_key_event(self, key: str, event_type: str):
        """Добавление события клавиши"""
        self.key_events.append(KeyEvent(
            key=key,
            event_type=event_type,
            timestamp=time.time()
        ))
    
    def calculate_features(self) -> Dict[str, float]:
        """Вычисление признаков из событий клавиш"""
        if len(self.key_events) < 2:
            return {}
        
        dwell_times = []  # Время удержания клавиш
        flight_times = []  # Время между клавишами
        
        # Группировка событий по клавишам
        key_press_times = {}
        key_release_times = {}
        
        for event in self.key_events:
            if event.event_type == 'press':
                key_press_times[event.key] = event.timestamp
            elif event.event_type == 'release':
                key_release_times[event.key] = event.timestamp
        
        # Вычисление времени удержания
        for key in key_press_times:
            if key in key_release_times:
                dwell_time = key_release_times[key] - key_press_times[key]
                dwell_times.append(dwell_time)
        
        # Вычисление времени между нажатиями
        press_events = sorted([e for e in self.key_events if e.event_type == 'press'], 
                            key=lambda x: x.timestamp)
        
        for i in range(1, len(press_events)):
            flight_time = press_events[i].timestamp - press_events[i-1].timestamp
            flight_times.append(flight_time)
        
        # Вычисление общей скорости печати
        if press_events:
            total_time = press_events[-1].timestamp - press_events[0].timestamp
            typing_speed = len(press_events) / total_time if total_time > 0 else 0
        else:
            typing_speed = 0
        
        # Формирование вектора признаков
        features = {
            'avg_dwell_time': sum(dwell_times) / len(dwell_times) if dwell_times else 0,
            'std_dwell_time': self._std(dwell_times) if len(dwell_times) > 1 else 0,
            'avg_flight_time': sum(flight_times) / len(flight_times) if flight_times else 0,
            'std_flight_time': self._std(flight_times) if len(flight_times) > 1 else 0,
            'typing_speed': typing_speed,
            'total_typing_time': total_time if press_events else 0
        }
        
        self.features = features
        return features
    
    def _std(self, values: List[float]) -> float:
        """Вычисление стандартного отклонения"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_feature_vector(self) -> List[float]:
        """Получение вектора признаков для ML"""
        if not self.features:
            self.calculate_features()
        
        return [
            self.features.get('avg_dwell_time', 0),
            self.features.get('std_dwell_time', 0),
            self.features.get('avg_flight_time', 0),
            self.features.get('std_flight_time', 0),
            self.features.get('typing_speed', 0),
            self.features.get('total_typing_time', 0)
        ]
    
    def save_raw_events_to_csv(self, user_id: int, username: str):
        """Сохранение сырых событий клавиш в CSV"""
        import csv
        import os
        from config import DATA_DIR
        
        # Создаем папку для CSV если её нет
        csv_dir = os.path.join(DATA_DIR, "csv_exports")
        os.makedirs(csv_dir, exist_ok=True)
        
        # Имя файла для сырых данных
        filename = f"user_{username}_raw_keystrokes.csv"
        filepath = os.path.join(csv_dir, filename)
        
        # Проверяем, существует ли файл
        file_exists = os.path.exists(filepath)
        
        try:
            # Открываем файл для добавления данных
            with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['session_id', 'timestamp', 'key', 'event_type', 'relative_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Записываем заголовок если файл новый
                if not file_exists:
                    writer.writeheader()
                
                # Записываем каждое событие
                if self.key_events:
                    start_time = self.key_events[0].timestamp
                    for event in self.key_events:
                        row = {
                            'session_id': self.session_id,
                            'timestamp': self.timestamp.isoformat(),
                            'key': event.key,
                            'event_type': event.event_type,
                            'relative_time': event.timestamp - start_time
                        }
                        writer.writerow(row)
        except PermissionError:
            # Если файл открыт в другой программе, просто пропускаем
            print(f"Предупреждение: не удалось записать в {filepath} - файл может быть открыт")