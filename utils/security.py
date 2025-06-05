# utils/security.py - Модуль безопасности

import hashlib
import secrets
from typing import Tuple

from config import SALT_LENGTH


class SecurityManager:
    """Менеджер безопасности для хеширования паролей"""
    
    def hash_password(self, password: str) -> Tuple[str, str]:
        """
        Хеширование пароля с солью
        
        Returns:
            Tuple[хеш_пароля, соль]
        """
        # Генерация соли
        salt = secrets.token_hex(SALT_LENGTH)
        
        # Хеширование пароля с солью
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        Проверка пароля
        
        Args:
            password: Введенный пароль
            password_hash: Хеш из БД
            salt: Соль из БД
            
        Returns:
            True если пароль верный
        """
        # Хеширование введенного пароля с солью из БД
        test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        
        # Сравнение хешей
        return test_hash == password_hash
    
    @staticmethod
    def generate_session_id() -> str:
        """Генерация уникального ID сессии"""
        return secrets.token_urlsafe(32)