# gui/login_window.py - Окно входа в систему

import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Callable, Optional

from models.user import User
from auth.password_auth import PasswordAuthenticator
from auth.keystroke_auth import KeystrokeAuthenticator
from config import FONT_FAMILY, FONT_SIZE, PANGRAM

class LoginWindow:
    """Окно входа с двухфакторной аутентификацией"""
    
    def __init__(self, parent, password_auth: PasswordAuthenticator, 
                 keystroke_auth: KeystrokeAuthenticator, on_success: Callable):
        self.parent = parent
        self.password_auth = password_auth
        self.keystroke_auth = keystroke_auth
        self.on_success = on_success
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title("Вход в систему")
        self.window.geometry("550x650")
        self.window.resizable(True, True)
        self.window.minsize(500, 600)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Центрирование
        self.center_window()
        
        # Переменные
        self.current_user: Optional[User] = None
        self.session_id: Optional[str] = None
        self.is_recording = False
        
        # Создание интерфейса
        self.create_widgets()
        
        # Фокус на поле ввода
        self.username_entry.focus()
    
    def center_window(self):
        """Центрирование окна"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Создание виджетов окна входа"""
        main_frame = ttk.Frame(self.window, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="Вход в систему",
            font=(FONT_FAMILY, 18, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Поля ввода
        # Имя пользователя
        ttk.Label(main_frame, text="Имя пользователя:").pack(anchor=tk.W, pady=(10, 5))
        self.username_entry = ttk.Entry(main_frame, width=30, font=(FONT_FAMILY, FONT_SIZE))
        self.username_entry.pack(fill=tk.X)
        
        # Пароль
        ttk.Label(main_frame, text="Пароль:").pack(anchor=tk.W, pady=(20, 5))
        self.password_entry = ttk.Entry(
            main_frame, 
            width=30, 
            show="*",
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.password_entry.pack(fill=tk.X)
        
        # Статус
        self.status_label = ttk.Label(
            main_frame,
            text="",
            foreground="blue"
        )
        self.status_label.pack(pady=10)
        
        # Информация о записи нажатий
        self.recording_frame = ttk.LabelFrame(
            main_frame,
            text="Проверка динамики нажатий",
            padding=10
        )
        self.recording_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        ttk.Label(
            self.recording_frame,
            text="После ввода пароля введите панграмму:",
            wraplength=300,
            justify=tk.CENTER
        ).pack()
        
        self.pangram_label = ttk.Label(
            self.recording_frame,
            text=f'"{PANGRAM}"',
            font=(FONT_FAMILY, 11, 'bold'),
            foreground='darkblue',
            wraplength=350
        )
        self.pangram_label.pack(pady=10)
        
        self.pangram_entry = ttk.Entry(
            self.recording_frame,
            width=45,
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.pangram_entry.pack(pady=5)
        
        self.recording_status = ttk.Label(
            self.recording_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.recording_status.pack(pady=5)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        self.login_btn = ttk.Button(
            button_frame,
            text="Войти",
            command=self.login,
            width=15
        )
        self.login_btn.grid(row=0, column=0, padx=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="Отмена",
            command=self.window.destroy,
            width=15
        )
        cancel_btn.grid(row=0, column=1, padx=5)
        
        # Привязка Enter к кнопке входа
        self.window.bind('<Return>', lambda e: self.login())
        
        # Обработчики для записи динамики
        self.setup_keystroke_recording()
        
        # Привязка обработчика для автоматического начала записи после ввода пароля
        self.password_entry.bind('<Return>', lambda e: self.pangram_entry.focus())
    
    def setup_keystroke_recording(self):
        """Настройка записи динамики нажатий для поля панграммы"""
        self.pangram_entry.bind('<FocusIn>', self.start_recording)
        self.pangram_entry.bind('<FocusOut>', self.stop_recording)
        self.pangram_entry.bind('<KeyPress>', self.on_key_press)
        self.pangram_entry.bind('<KeyRelease>', self.on_key_release)
    
    def start_recording(self, event=None):
        """Начало записи динамики"""
        if not self.is_recording and self.current_user:
            self.session_id = self.keystroke_auth.start_keystroke_recording(self.current_user.id)
            self.is_recording = True
            self.recording_status.config(
                text="🔴 Запись активна",
                foreground="red"
            )
    
    def stop_recording(self, event=None):
        """Остановка записи"""
        if self.is_recording:
            self.is_recording = False
            self.recording_status.config(
                text="",
                foreground="black"
            )
    
    def on_key_press(self, event):
        """Обработка нажатия клавиши"""
        if self.is_recording and self.session_id:
            # Игнорируем специальные клавиши
            if event.keysym not in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                                   'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab']:
                self.keystroke_auth.record_key_event(
                    self.session_id,
                    event.keysym,
                    'press'
                )
    
    def on_key_release(self, event):
        """Обработка отпускания клавиши"""
        if self.is_recording and self.session_id:
            if event.keysym not in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                                   'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab']:
                self.keystroke_auth.record_key_event(
                    self.session_id,
                    event.keysym,
                    'release'
                )
    
    def login(self):
        """Процесс входа в систему"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        pangram_text = self.pangram_entry.get()
        
        if not username or not password:
            messagebox.showerror("Ошибка", "Заполните имя пользователя и пароль")
            return
        
        # Первый фактор - проверка пароля
        self.status_label.config(text="Проверка пароля...", foreground="blue")
        self.window.update()
        
        success, message, user = self.password_auth.authenticate(username, password)
        
        if not success:
            self.status_label.config(text="", foreground="red")
            messagebox.showerror("Ошибка", message)
            self.password_entry.delete(0, tk.END)
            return
        
        self.current_user = user
        
        # Проверка, обучена ли модель
        if not user.is_trained:
            messagebox.showinfo(
                "Информация",
                "Модель динамики нажатий не обучена.\nВход выполнен только по паролю."
            )
            self.on_success(user)
            self.window.destroy()
            return
        
        # Проверка панграммы
        if pangram_text != PANGRAM:
            messagebox.showerror(
                "Ошибка",
                "Введите панграмму точно как показано для проверки динамики нажатий"
            )
            return
        
        # Второй фактор - проверка динамики нажатий
        if self.session_id:
            self.status_label.config(text="Анализ динамики нажатий...", foreground="blue")
            self.window.update()
            
            try:
                # Завершение записи и получение признаков
                features = self.keystroke_auth.finish_recording(self.session_id)
                
                # Аутентификация по динамике
                auth_success, confidence, auth_message = self.keystroke_auth.authenticate(
                    user, 
                    features
                )
                
                if auth_success:
                    self.status_label.config(
                        text=f"✓ {auth_message}",
                        foreground="green"
                    )
                    self.window.update()
                    time.sleep(1)  # Показать сообщение
                    
                    self.on_success(user)
                    self.window.destroy()
                else:
                    self.status_label.config(text="", foreground="red")
                    messagebox.showerror(
                        "Ошибка аутентификации",
                        f"{auth_message}\n\nДинамика набора не соответствует профилю пользователя."
                    )
                    self.password_entry.delete(0, tk.END)
                    self.pangram_entry.delete(0, tk.END)
                    self.current_user = None
                    
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при анализе динамики: {str(e)}")
                self.password_entry.delete(0, tk.END)
                self.pangram_entry.delete(0, tk.END)
        else:
            messagebox.showwarning(
                "Предупреждение",
                "Динамика нажатий не была записана.\nВведите панграмму для проверки."
            )