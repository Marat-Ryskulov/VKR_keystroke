# gui/training_window.py - Окно для обучения системы динамике нажатий

import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Callable

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from config import FONT_FAMILY, FONT_SIZE, MIN_TRAINING_SAMPLES, PANGRAM

class TrainingWindow:
    """Окно для обучения системы динамике нажатий пользователя"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator, on_complete: Callable):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        self.on_complete = on_complete
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title("Обучение системы")
        self.window.geometry("700x750")
        self.window.resizable(True, True)
        self.window.minsize(600, 700)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Центрирование
        self.center_window()
        
        # Переменные
        self.session_id = None
        self.is_recording = False
        self.current_sample = 0
        self.training_text = PANGRAM  # Используем панграмму из конфига
        
        # Создание интерфейса
        self.create_widgets()
        
        # Обновление прогресса
        self.update_progress()
    
    def center_window(self):
        """Центрирование окна"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Создание виджетов окна обучения"""
        main_frame = ttk.Frame(self.window, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="Обучение системы динамике нажатий",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Информация
        info_text = f"""Для обучения системы вашему уникальному стилю набора текста
необходимо {MIN_TRAINING_SAMPLES} раз ввести следующую панграмму:

"{PANGRAM}"

Старайтесь печатать естественно, как вы обычно это делаете.
Это займет примерно 10-15 минут."""
        
        info_label = ttk.Label(
            main_frame,
            text=info_text,
            wraplength=400,
            justify=tk.CENTER
        )
        info_label.pack(pady=10)
        
        # Прогресс
        progress_frame = ttk.LabelFrame(main_frame, text="Прогресс обучения", padding=15)
        progress_frame.pack(fill=tk.X, pady=20)
        
        self.progress_label = ttk.Label(
            progress_frame,
            text="",
            font=(FONT_FAMILY, 12)
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=400,
            mode='determinate',
            maximum=MIN_TRAINING_SAMPLES
        )
        self.progress_bar.pack(pady=10)
        
        # Поле ввода
        input_frame = ttk.LabelFrame(main_frame, text="Тренировочный ввод", padding=15)
        input_frame.pack(fill=tk.X, pady=10)
        
        self.pangram_label = ttk.Label(
            input_frame,
            text=f'Введите: "{PANGRAM}"',
            font=(FONT_FAMILY, 11, 'bold'),
            foreground='darkblue'
        )
        self.pangram_label.pack(pady=(0, 10))
        
        self.text_entry = ttk.Entry(
            input_frame,
            width=50,
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.text_entry.pack()
        
        self.status_label = ttk.Label(
            input_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.status_label.pack(pady=5)
        
        # Советы
        tips_frame = ttk.LabelFrame(main_frame, text="Советы", padding=10)
        tips_frame.pack(fill=tk.X, pady=10)
        
        tips_text = """• Печатайте в своем обычном темпе
• Не пытайтесь печатать идеально одинаково
• Расслабьтесь и печатайте естественно
• Если ошиблись - это нормально, продолжайте"""
        
        tips_label = ttk.Label(
            tips_frame,
            text=tips_text,
            justify=tk.LEFT,
            font=(FONT_FAMILY, 10)
        )
        tips_label.pack()
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        self.submit_btn = ttk.Button(
            button_frame,
            text="Сохранить образец",
            command=self.submit_sample,
            state=tk.DISABLED
        )
        self.submit_btn.grid(row=0, column=0, padx=5)
        
        self.train_btn = ttk.Button(
            button_frame,
            text="Обучить модель",
            command=self.train_model,
            state=tk.DISABLED
        )
        self.train_btn.grid(row=0, column=1, padx=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="Отмена",
            command=self.window.destroy
        )
        cancel_btn.grid(row=0, column=2, padx=5)
        
        # Привязка событий
        self.setup_keystroke_recording()
        self.text_entry.bind('<Return>', lambda e: self.submit_sample())
        
        # Фокус на поле ввода
        self.text_entry.focus()
    
    def setup_keystroke_recording(self):
        """Настройка записи динамики нажатий"""
        self.text_entry.bind('<FocusIn>', self.start_recording)
        self.text_entry.bind('<FocusOut>', self.stop_recording)
        self.text_entry.bind('<KeyPress>', self.on_key_press)
        self.text_entry.bind('<KeyRelease>', self.on_key_release)
        self.text_entry.bind('<KeyRelease>', self.check_input, add='+')
    
    def start_recording(self, event=None):
        """Начало записи"""
        if not self.is_recording:
            self.session_id = self.keystroke_auth.start_keystroke_recording(self.user.id)
            self.is_recording = True
            self.status_label.config(
                text="🔴 Запись активна",
                foreground="red"
            )
    
    def stop_recording(self, event=None):
        """Остановка записи"""
        if self.is_recording:
            self.is_recording = False
            self.status_label.config(
                text="Запись остановлена",
                foreground="gray"
            )
    
    def on_key_press(self, event):
        """Обработка нажатия клавиши"""
        if self.is_recording and self.session_id:
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
    
    def check_input(self, event=None):
        """Проверка готовности ввода"""
        # Проверяем, совпадает ли введенный текст с панграммой
        if self.text_entry.get() == PANGRAM:
            self.submit_btn.config(state=tk.NORMAL)
            self.status_label.config(
                text="✓ Текст введен правильно",
                foreground="green"
            )
        else:
            self.submit_btn.config(state=tk.DISABLED)
            if len(self.text_entry.get()) > 0:
                self.status_label.config(
                    text="Введите текст точно как показано выше",
                    foreground="orange"
                )
    
    def submit_sample(self):
        """Сохранение образца"""
        if self.text_entry.get() != PANGRAM:
            messagebox.showwarning("Предупреждение", "Введите панграмму точно как показано")
            return
        
        if self.session_id and self.is_recording:
            try:
                # Остановка записи
                self.stop_recording()
                
                # Завершение записи и сохранение
                features = self.keystroke_auth.finish_recording(self.session_id, is_training=True)
                
                # Обновление счетчика
                self.current_sample += 1
                
                # Сообщение об успехе
                self.status_label.config(
                    text=f"✓ Образец {self.current_sample} сохранен",
                    foreground="green"
                )
                
                # Очистка поля
                self.text_entry.delete(0, tk.END)
                
                # Обновление прогресса
                self.update_progress()
                
                # Фокус обратно на поле
                self.text_entry.focus()
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении образца: {str(e)}")
        else:
            messagebox.showwarning("Предупреждение", "Нет активной записи")
    
    def update_progress(self):
        """Обновление прогресса обучения"""
        progress = self.keystroke_auth.get_training_progress(self.user)
        
        self.current_sample = progress['current_samples']
        self.progress_label.config(
            text=f"Образцов собрано: {progress['current_samples']} из {progress['required_samples']}"
        )
        
        self.progress_bar['value'] = progress['current_samples']
        
        # Проверка готовности к обучению
        if progress['is_ready']:
            self.train_btn.config(state=tk.NORMAL)
            self.pangram_label.config(
                text=f"Достаточно образцов собрано! Можете обучить модель.",
                foreground="green"
            )
        else:
            remaining = progress['required_samples'] - progress['current_samples']
            self.pangram_label.config(
                text=f'Введите: "{PANGRAM}" (осталось {remaining} раз)',
                foreground="darkblue"
            )
    
    def train_model(self):
        """Обучение модели"""
        if messagebox.askyesno(
            "Подтверждение",
            "Начать обучение модели?\n\nЭто может занять несколько секунд."
        ):
            # Показываем прогресс
            self.train_btn.config(state=tk.DISABLED, text="Обучение...")
            self.window.update()
            
            try:
                # Обучение модели
                success, accuracy, message = self.keystroke_auth.train_user_model(self.user)
                
                if success:
                    messagebox.showinfo(
                        "Успех",
                        f"{message}\n\nТеперь вы можете использовать двухфакторную аутентификацию!"
                    )
                    self.on_complete()
                    self.window.destroy()
                else:
                    messagebox.showerror("Ошибка", message)
                    self.train_btn.config(state=tk.NORMAL, text="Обучить модель")
                    
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при обучении: {str(e)}")
                self.train_btn.config(state=tk.NORMAL, text="Обучить модель")