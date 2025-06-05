# models/user.py - Модель пользователя без full_name

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """Модель пользователя системы"""
    username: str
    password_hash: str
    salt: str
    id: Optional[int] = None
    is_trained: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    training_samples: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Преобразование в словарь для БД"""
        return {
            'id': self.id,
            'username': self.username,
            'password_hash': self.password_hash,
            'salt': self.salt,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_trained': int(self.is_trained),
            'training_samples': self.training_samples
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Создание из словаря БД"""
        return cls(
            id=data.get('id'),
            username=data['username'],
            password_hash=data['password_hash'],
            salt=data['salt'],
            is_trained=bool(data.get('is_trained', 0)),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None
        )