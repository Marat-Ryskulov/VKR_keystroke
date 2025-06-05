# auth/password_auth.py - Модуль парольной аутентификации без full_name

from typing import Tuple, Optional, List
from datetime import datetime

from models.user import User
from utils.database import DatabaseManager
from utils.security import SecurityManager


class PasswordAuthenticator:
    """Класс для парольной аутентификации без проверок надежности"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.security = SecurityManager()
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, str, Optional[User]]:
        """
        Аутентификация пользователя по паролю
        
        Returns:
            Tuple[успех, сообщение, пользователь]
        """
        # Получение пользователя из БД
        user = self.db.get_user_by_username(username)
        
        if not user:
            return False, "Пользователь не найден", None
        
        # Проверка пароля
        if not self.security.verify_password(password, user.password_hash, user.salt):
            return False, "Неверный пароль", None
        
        # Обновление времени последнего входа
        self.db.update_last_login(user.id)
        
        return True, "Успешная аутентификация", user
    
    def register(self, username: str, password: str) -> Tuple[bool, str, Optional[User]]:
        """
        Регистрация нового пользователя без проверок
        
        Returns:
            Tuple[успех, сообщение, пользователь]
        """
        # Проверка только на пустоту
        if not username or not password:
            return False, "Имя пользователя и пароль не могут быть пустыми", None
        
        # Проверка существования пользователя
        if self.db.get_user_by_username(username):
            return False, f"Пользователь '{username}' уже существует", None
        
        # Хеширование пароля
        password_hash, salt = self.security.hash_password(password)
        
        # Создание пользователя без full_name
        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt
        )
        
        # Сохранение в БД
        try:
            user_id = self.db.create_user(user)
            user.id = user_id
            return True, "Регистрация успешна", user
        except Exception as e:
            return False, f"Ошибка при регистрации: {str(e)}", None
    
    def change_password(self, user: User, old_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Изменение пароля пользователя без проверок
        
        Returns:
            Tuple[успех, сообщение]
        """
        # Проверка старого пароля
        if not self.security.verify_password(old_password, user.password_hash, user.salt):
            return False, "Неверный старый пароль"
        
        # Проверка только на пустоту
        if not new_password:
            return False, "Новый пароль не может быть пустым"
        
        # Хеширование нового пароля
        new_hash, new_salt = self.security.hash_password(new_password)
        
        # Обновление в БД
        try:
            self.db.update_user_password(user.id, new_hash, new_salt)
            user.password_hash = new_hash
            user.salt = new_salt
            return True, "Пароль успешно изменен"
        except Exception as e:
            return False, f"Ошибка при изменении пароля: {str(e)}"
    
    def get_all_users(self) -> List[User]:
        """Получение списка всех пользователей"""
        return self.db.get_all_users()
    
    def delete_user(self, user_id: int) -> bool:
        """Удаление пользователя"""
        try:
            self.db.delete_user(user_id)
            return True
        except:
            return False