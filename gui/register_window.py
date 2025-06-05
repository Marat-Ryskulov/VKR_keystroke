# gui/register_window.py - Окно регистрации без полного имени

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from models.user import User
from auth.password_auth import PasswordAuthenticator
from auth.keystroke_auth import KeystrokeAuthenticator
from utils.database import DatabaseManager
from config import FONT_FAMILY, FONT_SIZE


class RegisterWindow:
    """Окно регистрации нового пользователя"""
    
    def __init__(self, parent, password_auth: PasswordAuthenticator, 
                 keystroke_auth: KeystrokeAuthenticator, on_success: Callable):
        self.parent = parent
        self.password_auth = password_auth
        self.keystroke_auth = keystroke_auth
        self.on_success = on_success
        self.db = DatabaseManager()
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title("Регистрация")
        self.window.geometry("500x550")
        self.window.resizable(True, True)
        self.window.minsize(450, 800)
        
        # Запрет на изменение размера
        self.window.transient(parent)
        self.window.grab_set()
        
        # Создание виджетов
        self.create_widgets()
        
        # Центрирование окна
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Фокус на первом поле
        self.username_entry.focus()
    
    def create_widgets(self):
        """Создание виджетов окна регистрации"""
        # Основной фрейм
        main_frame = ttk.Frame(self.window, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="Создание нового аккаунта",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Фрейм для имени пользователя
        username_frame = ttk.LabelFrame(main_frame, text="Данные пользователя", padding=15)
        username_frame.pack(fill=tk.X, pady=10)
        
        # Поле имени пользователя
        ttk.Label(
            username_frame,
            text="Имя пользователя:",
            font=(FONT_FAMILY, FONT_SIZE)
        ).pack(anchor=tk.W)
        
        self.username_entry = ttk.Entry(
            username_frame,
            width=30,
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.username_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Подсказка
        ttk.Label(
            username_frame,
            text="Можно использовать любые символы",
            font=(FONT_FAMILY, 9),
            foreground="gray"
        ).pack(anchor=tk.W)
        
        # Фрейм для пароля
        password_frame = ttk.LabelFrame(main_frame, text="Безопасность", padding=15)
        password_frame.pack(fill=tk.X, pady=10)
        
        # Поле пароля
        ttk.Label(
            password_frame,
            text="Пароль:",
            font=(FONT_FAMILY, FONT_SIZE)
        ).pack(anchor=tk.W)
        
        self.password_entry = ttk.Entry(
            password_frame,
            width=30,
            show="*",
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.password_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Подтверждение пароля
        ttk.Label(
            password_frame,
            text="Подтвердите пароль:",
            font=(FONT_FAMILY, FONT_SIZE)
        ).pack(anchor=tk.W, pady=(15, 0))
        
        self.password_confirm_entry = ttk.Entry(
            password_frame,
            width=30,
            show="*",
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.password_confirm_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Информация о пароле
        req_frame = ttk.LabelFrame(password_frame, text="Информация", padding=10)
        req_frame.pack(fill=tk.X, pady=(10, 0))
        
        requirements = """• Можно использовать любой пароль
• Пароль не должен быть пустым
• Запомните пароль - восстановление невозможно"""
        
        req_label = ttk.Label(
            req_frame,
            text=requirements,
            font=(FONT_FAMILY, 9),
            foreground="darkblue"
        )
        req_label.pack(anchor=tk.W)
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # Кнопка регистрации
        self.register_btn = ttk.Button(
            button_frame,
            text="Создать аккаунт",
            command=self.register,
            style='Accent.TButton'
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка отмены
        cancel_btn = ttk.Button(
            button_frame,
            text="Отмена",
            command=self.window.destroy
        )
        cancel_btn.pack(side=tk.LEFT)
        
        # Статус
        self.status_label = ttk.Label(
            main_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.status_label.pack()
        
        # Обработчики событий
        self.username_entry.bind('<Return>', lambda e: self.password_entry.focus())
        self.password_entry.bind('<Return>', lambda e: self.password_confirm_entry.focus())
        self.password_confirm_entry.bind('<Return>', lambda e: self.register())
    
    def validate_input(self):
        """Валидация введенных данных"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        password_confirm = self.password_confirm_entry.get()
        
        # Проверка, что поля не пустые
        if not username:
            messagebox.showerror("Ошибка", "Введите имя пользователя")
            return False
        
        if not password:
            messagebox.showerror("Ошибка", "Введите пароль")
            return False
        
        # Проверка совпадения паролей
        if password != password_confirm:
            messagebox.showerror("Ошибка", "Пароли не совпадают")
            return False
        
        # Проверка существования пользователя
        if self.db.get_user_by_username(username):
            messagebox.showerror("Ошибка", f"Пользователь '{username}' уже существует")
            return False
        
        return True
    
    def register(self):
        """Регистрация пользователя"""
        if not self.validate_input():
            return
        
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        # Обновление статуса
        self.status_label.config(text="Создание аккаунта...", foreground="blue")
        self.register_btn.config(state=tk.DISABLED)
        self.window.update()
        
        # Регистрация (убрали full_name)
        success, message, user = self.password_auth.register(username, password)
        
        if success:
            self.status_label.config(text="Успешная регистрация!", foreground="green")
            messagebox.showinfo(
                "Успех",
                f"Аккаунт '{username}' успешно создан!\n\n"
                "Теперь вам необходимо обучить систему вашему стилю набора текста."
            )
            self.on_success(user)
            self.window.destroy()
        else:
            self.status_label.config(text="", foreground="red")
            self.register_btn.config(state=tk.NORMAL)
            messagebox.showerror("Ошибка", message)