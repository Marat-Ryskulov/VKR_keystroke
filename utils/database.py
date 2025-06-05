# utils/database.py - Утилиты для работы с базой данных без full_name

import sqlite3
import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from models.user import User
from models.keystroke_data import KeystrokeData
from config import DATABASE_PATH, DATA_DIR


class DatabaseManager:
    """Менеджер базы данных"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self._create_tables()
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для подключения к БД"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _create_tables(self):
        """Создание таблиц в БД"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT,
                    is_trained INTEGER DEFAULT 0,
                    training_samples INTEGER DEFAULT 0
                )
            ''')
            
            # Таблица образцов клавиатурного почерка
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keystroke_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_training INTEGER DEFAULT 1,
                    features TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            
            # Индексы для производительности
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users (username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_samples ON keystroke_samples (user_id)')
    
    def create_user(self, user: User) -> Optional[int]:
        """Создание нового пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try: 
                cursor.execute('''
                    INSERT INTO users (username, password_hash, salt, created_at, is_trained, training_samples)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user.username,
                    user.password_hash,
                    user.salt,
                    user.created_at.isoformat() if user.created_at else None,
                    int(user.is_trained),
                    user.training_samples
                ))
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                print(f"Ошибка при добавлении пользователя: {e}")
                return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Получение пользователя по имени"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            
            if row:
                return User.from_dict(dict(row))
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Получение пользователя по ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            
            if row:
                return User.from_dict(dict(row))
            return None
    
    def get_all_users(self) -> List[User]:
        """Получение списка всех пользователей"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users ORDER BY username')
            
            users = []
            for row in cursor.fetchall():
                users.append(User.from_dict(dict(row)))
            return users
    
    def update_user_password(self, user_id: int, new_password_hash: str, new_salt: str):
        """Обновление пароля пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET password_hash = ?, salt = ?
                WHERE id = ?
            ''', (new_password_hash, new_salt, user_id))
    
    def update_user_trained_status(self, user_id: int, is_trained: bool):
        """Обновление статуса обучения пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET is_trained = ?
                WHERE id = ?
            ''', (int(is_trained), user_id))
    
    def update_last_login(self, user_id: int):
        """Обновление времени последнего входа"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))
    
    def save_keystroke_sample(self, keystroke_data: KeystrokeData, is_training: bool = True):
        """Сохранение образца клавиатурного почерка"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Сериализация признаков в JSON
            features_json = json.dumps(keystroke_data.features)
            
            cursor.execute('''
                INSERT INTO keystroke_samples (user_id, session_id, timestamp, features, is_training)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                keystroke_data.user_id,
                keystroke_data.session_id,
                keystroke_data.timestamp.isoformat(),
                features_json,
                int(is_training)
            ))
            
            # Сохранение в CSV файл
            if is_training:
                self._save_to_csv(keystroke_data, is_training)
    
    def get_user_keystroke_samples(self, user_id: int, training_only: bool = True) -> List[KeystrokeData]:
        """Получение образцов клавиатурного почерка пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM keystroke_samples 
                WHERE user_id = ?
            '''
            
            if training_only:
                query += ' AND is_training = 1'
            
            query += ' ORDER BY timestamp'
            
            cursor.execute(query, (user_id,))
            
            samples = []
            for row in cursor.fetchall():
                keystroke_data = KeystrokeData(
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )
                keystroke_data.features = json.loads(row['features'])
                samples.append(keystroke_data)
            
            return samples
    
    def get_user_training_samples(self, user_id: int) -> List[KeystrokeData]:
        """Получение всех обучающих образцов пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM keystroke_samples 
                WHERE user_id = ? AND is_training = 1
                ORDER BY timestamp
            ''', (user_id,))
            
            samples = []
            for row in cursor.fetchall():
                keystroke_data = KeystrokeData(
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )
                keystroke_data.features = json.loads(row['features'])
                samples.append(keystroke_data)
            
            return samples
    
    def delete_user_samples(self, user_id: int):
        """Удаление всех образцов пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM keystroke_samples WHERE user_id = ?', (user_id,))
    
    def delete_user(self, user_id: int):
        """Удаление пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    
    def _save_to_csv(self, keystroke_data: KeystrokeData, is_training: bool):
        """Сохранение данных в CSV файл"""
        import csv
        
        # Создаем папку для CSV если её нет
        csv_dir = os.path.join(DATA_DIR, "csv_exports")
        os.makedirs(csv_dir, exist_ok=True)
        
        # Получаем имя пользователя
        user = self.get_user_by_id(keystroke_data.user_id)
        if not user:
            return
        
        # Имя файла
        filename = f"user_{user.username}_keystroke_data.csv"
        filepath = os.path.join(csv_dir, filename)
        
        # Проверяем, существует ли файл
        file_exists = os.path.exists(filepath)
        
        try:
            # Открываем файл для добавления данных
            with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'session_id', 'is_training',
                    'avg_dwell_time', 'std_dwell_time',
                    'avg_flight_time', 'std_flight_time',
                    'typing_speed', 'total_typing_time'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Записываем заголовок если файл новый
                if not file_exists:
                    writer.writeheader()
                
                # Записываем данные
                row = {
                    'timestamp': keystroke_data.timestamp.isoformat(),
                    'session_id': keystroke_data.session_id,
                    'is_training': is_training,
                    **keystroke_data.features
                }
                writer.writerow(row)
        except PermissionError:
            # Если файл открыт в другой программе, просто пропускаем
            print(f"Предупреждение: не удалось записать в {filepath} - файл может быть открыт")