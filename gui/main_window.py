# gui/main_window.py - Главное окно приложения

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from gui.login_window import LoginWindow
from gui.register_window import RegisterWindow
from models.user import User
from auth.password_auth import PasswordAuthenticator
from auth.keystroke_auth import KeystrokeAuthenticator
import config
from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT, FONT_FAMILY, FONT_SIZE

class MainWindow:
    """Главное окно приложения"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(True, True)
        self.root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Центрирование окна
        self.center_window()
        
        # Инициализация компонентов
        self.password_auth = PasswordAuthenticator()
        self.keystroke_auth = KeystrokeAuthenticator()
        
        # Текущий пользователь
        self.current_user: Optional[User] = None
        
        # Стили
        self.setup_styles()
        
        # Создание интерфейса
        self.create_widgets()
    
    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настройка цветов
        style.configure('Title.TLabel', font=(FONT_FAMILY, 20, 'bold'))
        style.configure('Header.TLabel', font=(FONT_FAMILY, 14, 'bold'))
        style.configure('Info.TLabel', font=(FONT_FAMILY, FONT_SIZE))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Big.TButton', font=(FONT_FAMILY, 12), padding=10)
    
    def create_widgets(self):
        """Создание виджетов главного окна"""
        # Заголовок
        title_frame = ttk.Frame(self.root, padding=20)
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            title_frame,
            text="Двухфакторная аутентификация",
            style='Title.TLabel'
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="с использованием динамики нажатий клавиш",
            style='Info.TLabel'
        )
        subtitle_label.pack()
        
        # Основная область
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Показываем начальный экран
        self.show_welcome_screen()
    
    def show_welcome_screen(self):
        """Отображение экрана приветствия"""
        self.clear_main_frame()
        
        welcome_frame = ttk.Frame(self.main_frame)
        welcome_frame.pack(expand=True)
        
        # Информация о системе
        info_text = """
        Эта система использует два фактора аутентификации:
        1. Традиционный пароль
        2. Уникальный стиль набора текста
        
        Система анализирует:
        • Время удержания клавиш
        • Время между нажатиями
        • Общий ритм печати
        """
        
        info_label = ttk.Label(
            welcome_frame,
            text=info_text,
            style='Info.TLabel',
            justify=tk.LEFT
        )
        info_label.pack(pady=20)
        
        # Кнопки
        button_frame = ttk.Frame(welcome_frame)
        button_frame.pack(pady=20)
        
        login_btn = ttk.Button(
            button_frame,
            text="Войти",
            style='Big.TButton',
            command=self.show_login
        )
        login_btn.grid(row=0, column=0, padx=10)
        
        register_btn = ttk.Button(
            button_frame,
            text="Регистрация",
            style='Big.TButton',
            command=self.show_register
        )
        register_btn.grid(row=0, column=1, padx=10)
    
    def show_login(self):
        """Показать окно входа"""
        login_window = LoginWindow(
            self.root,
            self.password_auth,
            self.keystroke_auth,
            self.on_login_success
        )
    
    def show_register(self):
        """Показать окно регистрации"""
        register_window = RegisterWindow(
            self.root,
            self.password_auth,
            self.keystroke_auth,
            self.on_register_success
        )
    
    def on_login_success(self, user: User):
        """Обработка успешного входа"""
        self.current_user = user
        self.show_user_dashboard()
    
    def show_user_dashboard(self):
        """Отображение панели пользователя"""
        self.clear_main_frame()
        
        # Заголовок с именем пользователя
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        welcome_label = ttk.Label(
            header_frame,
            text=f"Добро пожаловать, {self.current_user.username}!",
            style='Header.TLabel'
        )
        welcome_label.pack()
        
        # Информация о статусе обучения
        status_frame = ttk.LabelFrame(self.main_frame, text="Статус системы", padding=15)
        status_frame.pack(fill=tk.X, pady=10)
        
        training_progress = self.keystroke_auth.get_training_progress(self.current_user)
        
        if self.current_user.is_trained:
            status_text = "✓ Модель обучена и готова к использованию"
            status_style = 'Success.TLabel'
        else:
            status_text = f"⚠ Требуется обучение ({training_progress['current_samples']}/{training_progress['required_samples']} образцов)"
            status_style = 'Error.TLabel'
        
        status_label = ttk.Label(status_frame, text=status_text, style=status_style)
        status_label.pack()
        
        # Прогресс-бар для обучения
        if not self.current_user.is_trained:
            progress_bar = ttk.Progressbar(
                status_frame,
                value=training_progress['progress_percent'],
                maximum=100,
                length=300
            )
            progress_bar.pack(pady=10)
        
        # Статистика
        stats_frame = ttk.LabelFrame(self.main_frame, text="Статистика", padding=15)
        stats_frame.pack(fill=tk.X, pady=10)
        
        auth_stats = self.keystroke_auth.get_authentication_stats(self.current_user)
        
        stats_text = f"""
        Обучающих образцов: {auth_stats['training_samples']}
        Попыток аутентификации: {auth_stats['authentication_attempts']}
        Дата регистрации: {self.current_user.created_at.strftime('%d.%m.%Y')}
        """
        
        stats_label = ttk.Label(stats_frame, text=stats_text.strip(), justify=tk.LEFT)
        stats_label.pack()
        
        # Кнопки действий
        actions_frame = ttk.Frame(self.main_frame)
        actions_frame.pack(fill=tk.X, pady=20)
        
        if not self.current_user.is_trained:
            train_btn = ttk.Button(
                actions_frame,
                text="Начать обучение",
                command=self.start_training,
                style='Big.TButton'
            )
            train_btn.pack(pady=5)
        else:
            test_btn = ttk.Button(
                actions_frame,
                text="Тест аутентификации",
                command=self.test_authentication,
                style='Big.TButton'
            )
            test_btn.pack(pady=5)
            
            stats_btn = ttk.Button(
                actions_frame,
                text="Статистика модели",
                command=self.show_model_stats,
                style='Big.TButton'
            )
            stats_btn.pack(pady=5)
            
            retrain_btn = ttk.Button(
                actions_frame,
                text="Переобучить модель",
                command=self.reset_and_retrain
            )
            retrain_btn.pack(pady=5)
        
        # Кнопка экспорта данных
        export_btn = ttk.Button(
            actions_frame,
            text="Открыть папку с CSV файлами",
            command=self.open_csv_folder
        )
        export_btn.pack(pady=5)
        
        logout_btn = ttk.Button(
            actions_frame,
            text="Выйти",
            command=self.logout
        )
        logout_btn.pack(pady=5)
    
    def start_training(self):
        """Начать процесс обучения"""
        from gui.training_window import TrainingWindow
        TrainingWindow(
            self.root,
            self.current_user,
            self.keystroke_auth,
            self.on_training_complete
        )
    
    def test_authentication(self):
        """Тестирование аутентификации"""
        self.logout()
        self.show_login()
    
    def reset_and_retrain(self):
        """Сброс и переобучение модели"""
        if messagebox.askyesno(
            "Подтверждение",
            "Вы уверены, что хотите сбросить модель и начать обучение заново?"
        ):
            success, message = self.keystroke_auth.reset_user_model(self.current_user)
            if success:
                self.current_user.is_trained = False
                self.show_user_dashboard()
            else:
                messagebox.showerror("Ошибка", message)
    
    def on_training_complete(self):
        """Обработка завершения обучения"""
        # Обновляем информацию о пользователе
        self.current_user = self.password_auth.db.get_user_by_username(self.current_user.username)
        self.show_user_dashboard()
    
    def show_model_stats(self):
        """Показать статистику модели"""
        from gui.model_stats_window import ModelStatsWindow
        ModelStatsWindow(self.root, self.current_user)
    
    def open_csv_folder(self):
        """Открытие папки с CSV файлами"""
        import os
        import subprocess
        import platform
        
        csv_dir = os.path.join(config.DATA_DIR, "csv_exports")
        
        # Создаем папку если её нет
        os.makedirs(csv_dir, exist_ok=True)
        
        # Открываем папку в проводнике
        if platform.system() == 'Windows':
            os.startfile(csv_dir)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.Popen(['open', csv_dir])
        else:  # Linux
            subprocess.Popen(['xdg-open', csv_dir])
        
        messagebox.showinfo(
            "CSV файлы",
            f"Папка с CSV файлами открыта.\n\nПуть: {csv_dir}\n\n"
            "Файлы:\n"
            "• user_[имя]_keystroke_data.csv - статистика по образцам\n"
            "• user_[имя]_raw_keystrokes.csv - сырые данные нажатий"
        )
    
    def logout(self):
        """Выход из системы"""
        self.current_user = None
        self.show_welcome_screen()
    
    def clear_main_frame(self):
        """Очистка основного фрейма"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()
    
    def on_register_success(self, user: User):
        """Обработка успешной регистрации"""
        messagebox.showinfo(
            "Успех",
            "Регистрация завершена!\nТеперь необходимо пройти обучение системы."
        )
        self.current_user = user
        self.show_user_dashboard()