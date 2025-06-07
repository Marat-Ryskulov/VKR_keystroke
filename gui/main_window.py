# gui/main_window.py - Адаптивное главное окно для 1920x1080

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
    """Адаптивное главное окно приложения"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        
        # Адаптивные размеры
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(True, True)
        self.root.minsize(600, 700)
        self.root.maxsize(1200, 1200)
        
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
        """Настройка адаптивных стилей"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Адаптивные размеры шрифтов
        title_size = max(16, FONT_SIZE + 6)
        header_size = max(12, FONT_SIZE + 2)
        button_size = max(10, FONT_SIZE)
        
        style.configure('Title.TLabel', font=(FONT_FAMILY, title_size, 'bold'))
        style.configure('Header.TLabel', font=(FONT_FAMILY, header_size, 'bold'))
        style.configure('Info.TLabel', font=(FONT_FAMILY, FONT_SIZE))
        style.configure('Success.TLabel', foreground='green', font=(FONT_FAMILY, FONT_SIZE))
        style.configure('Error.TLabel', foreground='red', font=(FONT_FAMILY, FONT_SIZE))
        style.configure('Big.TButton', font=(FONT_FAMILY, button_size), padding=(10, 8))
        style.configure('Compact.TButton', font=(FONT_FAMILY, FONT_SIZE-1), padding=(8, 4))
    
    def create_widgets(self):
        """Создание адаптивных виджетов"""
        # Главный контейнер с прокруткой
        self.create_scrollable_container()
        
        # Заголовок
        self.create_header()
        
        # Основная область
        self.main_frame = ttk.Frame(self.scrollable_frame, padding=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Показываем начальный экран
        self.show_welcome_screen()
    
    def create_scrollable_container(self):
        """Создание контейнера с прокруткой"""
        # Canvas для прокрутки
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Привязка колесика мыши
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
    
    def create_header(self):
        """Создание компактного заголовка"""
        header_frame = ttk.Frame(self.scrollable_frame, padding=15)
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame,
            text="Двухфакторная аутентификация",
            style='Title.TLabel'
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            header_frame,
            text="с использованием динамики нажатий клавиш",
            style='Info.TLabel'
        )
        subtitle_label.pack()
    
    def show_welcome_screen(self):
        """Компактный экран приветствия"""
        self.clear_main_frame()
        
        welcome_frame = ttk.Frame(self.main_frame)
        welcome_frame.pack(fill=tk.BOTH, expand=True)
        
        # Компактная информация о системе
        info_text = """Система использует два фактора аутентификации:
1. Традиционный пароль
2. Уникальный стиль набора текста

Анализируемые параметры:
• Время удержания клавиш
• Время между нажатиями  
• Общий ритм печати"""
        
        info_label = ttk.Label(
            welcome_frame,
            text=info_text,
            style='Info.TLabel',
            justify=tk.LEFT
        )
        info_label.pack(pady=15)
        
        # Кнопки в одну строку
        button_frame = ttk.Frame(welcome_frame)
        button_frame.pack(pady=15)
        
        login_btn = ttk.Button(
            button_frame,
            text="Войти",
            style='Big.TButton',
            command=self.show_login
        )
        login_btn.pack(side=tk.LEFT, padx=10)
        
        register_btn = ttk.Button(
            button_frame,
            text="Регистрация", 
            style='Big.TButton',
            command=self.show_register
        )
        register_btn.pack(side=tk.LEFT, padx=10)
    
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
        """Компактная панель пользователя"""
        # 🔍 ОТЛАДКА
        print(f"\n=== ОТЛАДКА для пользователя {self.current_user.username} ===")
        try:
            self.password_auth.db.debug_user_samples(self.current_user.id)
        except AttributeError:
            print("Метод debug_user_samples не найден")

        self.clear_main_frame()
    
        # Компактный заголовок
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
    
        welcome_label = ttk.Label(
            header_frame,
            text=f"Добро пожаловать, {self.current_user.username}!",
            style='Header.TLabel'
        )
        welcome_label.pack()
    
        # Статус в компактном виде
        status_frame = ttk.LabelFrame(self.main_frame, text="Статус системы", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
    
        training_progress = self.keystroke_auth.get_training_progress(self.current_user)
    
        if self.current_user.is_trained:
            status_text = "✅ Модель обучена и готова к использованию"
            status_style = 'Success.TLabel'
        else:
            status_text = f"⚠️ Требуется обучение ({training_progress['current_samples']}/{training_progress['required_samples']} образцов)"
            status_style = 'Error.TLabel'
    
        status_label = ttk.Label(status_frame, text=status_text, style=status_style)
        status_label.pack()
    
        # Компактный прогресс-бар
        if not self.current_user.is_trained:
            progress_bar = ttk.Progressbar(
                status_frame,
                value=training_progress['progress_percent'],
                maximum=100,
                length=400
            )
            progress_bar.pack(pady=8)
    
        # Компактная статистика
        stats_frame = ttk.LabelFrame(self.main_frame, text="Статистика", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
    
        try:
            training_samples = self.password_auth.db.get_user_training_samples(self.current_user.id)
            training_samples_count = len(training_samples)
            auth_attempts = self.password_auth.db.get_auth_attempts(self.current_user.id, limit=50)
            auth_attempts_count = len(auth_attempts)
        
            # Горизонтальная статистика
            stats_text = f"Обучающих: {training_samples_count} | Попыток входа: {auth_attempts_count} | Регистрация: {self.current_user.created_at.strftime('%d.%m.%Y')}"
        
            stats_label = ttk.Label(stats_frame, text=stats_text, style='Info.TLabel')
            stats_label.pack()
        
        except Exception as e:
            error_label = ttk.Label(stats_frame, text=f"Ошибка статистики: {str(e)}", style='Error.TLabel')
            error_label.pack()
    
        # ОСНОВНЫЕ КНОПКИ ДЕЙСТВИЙ - всегда видны
        actions_frame = ttk.LabelFrame(self.main_frame, text="Действия", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
    
        # Компактная сетка кнопок 2x3
        if not self.current_user.is_trained:
            # Кнопка обучения - самая важная
            train_btn = ttk.Button(
                actions_frame,
                text="🎓 Начать обучение",
                style='Big.TButton',
                command=self.start_training
            )
            train_btn.pack(fill=tk.X, pady=3)
        else:
            # Сетка кнопок для обученной модели
            buttons_grid = ttk.Frame(actions_frame)
            buttons_grid.pack(fill=tk.X)
            
            # Ряд 1
            row1 = ttk.Frame(buttons_grid)
            row1.pack(fill=tk.X, pady=2)
            
            test_btn = ttk.Button(
                row1,
                text="🔐 Тест входа",
                style='Compact.TButton',
                command=self.test_authentication
            )
            test_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
            
            stats_btn = ttk.Button(
                row1,
                text="📊 Статистика",
                style='Compact.TButton',
                command=self.show_model_stats
            )
            stats_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            
            enhanced_btn = ttk.Button(
                row1,
                text="📈 Продвинутая",
                style='Compact.TButton',
                command=self.show_enhanced_stats
            )
            enhanced_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
            
            # Ряд 2
            row2 = ttk.Frame(buttons_grid)
            row2.pack(fill=tk.X, pady=2)
            
            retrain_btn = ttk.Button(
                row2,
                text="🔄 Переобучить",
                style='Compact.TButton',
                command=self.reset_and_retrain
            )
            retrain_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
            
            csv_btn = ttk.Button(
                row2,
                text="📁 CSV файлы",
                style='Compact.TButton',
                command=self.open_csv_folder
            )
            csv_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            
            logout_btn = ttk.Button(
                row2,
                text="🚪 Выйти",
                style='Compact.TButton',
                command=self.logout
            )
            logout_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        
        # Кнопки общего назначения
        general_frame = ttk.Frame(self.main_frame)
        general_frame.pack(fill=tk.X, pady=5)
        
        if self.current_user.is_trained:
            pass  # Кнопки уже добавлены выше
        else:
            # Дополнительные кнопки для необученной модели
            extra_row = ttk.Frame(general_frame)
            extra_row.pack(fill=tk.X)
            
            csv_btn = ttk.Button(
                extra_row,
                text="📁 CSV файлы",
                style='Compact.TButton',
                command=self.open_csv_folder
            )
            csv_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
            
            logout_btn = ttk.Button(
                extra_row,
                text="🚪 Выйти",
                style='Compact.TButton',
                command=self.logout
            )
            logout_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
    
    def start_training(self):
        """Начать процесс обучения"""
        choice = messagebox.askyesnocancel(
            "Выбор метода обучения",
            "Выберите метод обучения:\n\n"
            "ДА - Продвинутое обучение (с валидацией)\n"
            "НЕТ - Базовое обучение (быстрое)\n"
            "ОТМЕНА - Отменить"
        )
    
        if choice is None:
            return
        elif choice:
            try:
                from gui.enhanced_training_window import EnhancedTrainingWindow
                EnhancedTrainingWindow(
                    self.root,
                    self.current_user,
                    self.keystroke_auth,
                    self.on_training_complete
                )
            except ImportError:
                messagebox.showerror("Ошибка", "Модуль продвинутого обучения не найден.")
                from gui.training_window import TrainingWindow
                TrainingWindow(
                    self.root,
                    self.current_user,
                    self.keystroke_auth,
                    self.on_training_complete
                )
        else:
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
            "Вы уверены, что хотите сбросить модель и начать обучение заново?\n\nВсе обучающие данные будут удалены!"
        ):
            success, message = self.keystroke_auth.reset_user_model(self.current_user)
            if success:
                self.current_user.is_trained = False
                messagebox.showinfo("Успех", "Модель сброшена. Теперь можете начать обучение заново.")
                self.show_user_dashboard()
            else:
                messagebox.showerror("Ошибка", message)
    
    def on_training_complete(self):
        """Обработка завершения обучения"""
        updated_user = self.password_auth.db.get_user_by_username(self.current_user.username)
        if updated_user:
            self.current_user = updated_user
        self.show_user_dashboard()
    
    def show_model_stats(self):
        """Показать базовую статистику модели"""
        if not self.current_user or not self.current_user.is_trained:
            messagebox.showwarning("Предупреждение", "Модель не обучена")
            return
    
        try:
            from gui.model_stats_window import ModelStatsWindow
            ModelStatsWindow(self.root, self.current_user, self.keystroke_auth)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка статистики: {str(e)}")
    
    def show_enhanced_stats(self):
        """Показать продвинутую статистику модели"""
        if not self.current_user or not self.current_user.is_trained:
            messagebox.showwarning("Предупреждение", "Модель не обучена")
            return
    
        try:
            from gui.enhanced_model_stats_window import EnhancedModelStatsWindow
            EnhancedModelStatsWindow(self.root, self.current_user, self.keystroke_auth)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка продвинутой статистики: {str(e)}")
    
    def open_csv_folder(self):
        """Открытие папки с CSV файлами"""
        import os
        import subprocess
        import platform
        
        csv_dir = os.path.join(config.DATA_DIR, "csv_exports")
        os.makedirs(csv_dir, exist_ok=True)
        
        try:
            if platform.system() == 'Windows':
                os.startfile(csv_dir)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', csv_dir])
            else:
                subprocess.Popen(['xdg-open', csv_dir])
            
            messagebox.showinfo("CSV файлы", f"Папка открыта: {csv_dir}")
        except Exception as e:
            messagebox.showwarning("Предупреждение", f"Не удалось открыть папку: {e}")
    
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