# gui/enhanced_training_window.py - Адаптивное окно обучения для 1920x1080

import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
from typing import Callable

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from config import TRAINING_WINDOW_WIDTH, TRAINING_WINDOW_HEIGHT, FONT_FAMILY, FONT_SIZE, MIN_TRAINING_SAMPLES, PANGRAM

class EnhancedTrainingWindow:
    """Адаптивное окно для обучения с выбором метода валидации"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator, on_complete: Callable):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        self.on_complete = on_complete
        
        # Создание адаптивного окна
        self.window = tk.Toplevel(parent)
        self.window.title("🚀 Продвинутое обучение системы")
        self.window.geometry(f"{TRAINING_WINDOW_WIDTH}x{TRAINING_WINDOW_HEIGHT}")
        self.window.resizable(True, True)
        self.window.minsize(700, 800)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Центрирование
        self.center_window()
        
        # Переменные
        self.session_id = None
        self.is_recording = False
        self.current_sample = 0
        self.training_text = PANGRAM
        self.use_enhanced_training = tk.BooleanVar(value=True)
        self.training_in_progress = False
        
        # Нормализованный текст для сравнения
        self.normalized_target = self._normalize_text(PANGRAM)
        
        # Создание прокручиваемого интерфейса
        self.create_scrollable_interface()
        
        # Обновление прогресса
        self.update_progress()
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        return text.lower().replace(" ", "")
    
    def center_window(self):
        """Центрирование окна"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_scrollable_interface(self):
        """Создание прокручиваемого интерфейса"""
        # Главный контейнер с прокруткой
        main_canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = ttk.Frame(main_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязка колесика мыши
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Создание содержимого
        self.create_widgets()
    
    def create_widgets(self):
        """Создание адаптивных виджетов"""
        main_frame = ttk.Frame(self.scrollable_frame, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Компактный заголовок
        title_label = ttk.Label(
            main_frame,
            text="🚀 Продвинутое обучение системы",
            font=(FONT_FAMILY, FONT_SIZE + 4, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Выбор метода обучения - компактно
        method_frame = ttk.LabelFrame(main_frame, text="⚙️ Метод обучения", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        enhanced_radio = ttk.Radiobutton(
            method_frame,
            text="🔬 Продвинутое обучение (с кросс-валидацией)",
            variable=self.use_enhanced_training,
            value=True
        )
        enhanced_radio.pack(anchor=tk.W)
        
        basic_radio = ttk.Radiobutton(
            method_frame,
            text="⚡ Базовое обучение (быстрое)",
            variable=self.use_enhanced_training,
            value=False
        )
        basic_radio.pack(anchor=tk.W)
        
        # Компактное описание
        desc_label = ttk.Label(
            method_frame,
            text="🔬 Продвинутое: кросс-валидация + оптимизация\n⚡ Базовое: простое обучение без валидации",
            font=(FONT_FAMILY, FONT_SIZE-1),
            foreground="gray"
        )
        desc_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Информация о процессе - компактно
        info_frame = ttk.LabelFrame(main_frame, text="📋 Процесс обучения", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = f"""Для обучения необходимо {MIN_TRAINING_SAMPLES} образцов вашего стиля печати
Панграмма: "{PANGRAM}"

ПРАВИЛА:
• Печатайте в обычном темпе - НЕ торопитесь
• Регистр букв не важен, пробелы можно пропускать
• При ошибке ввод сбросится автоматически"""
        
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            font=(FONT_FAMILY, FONT_SIZE-1),
            justify=tk.LEFT
        )
        info_label.pack()
        
        # Компактный прогресс
        progress_frame = ttk.LabelFrame(main_frame, text="📊 Прогресс", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(
            progress_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=500,
            mode='determinate',
            maximum=MIN_TRAINING_SAMPLES
        )
        self.progress_bar.pack(pady=5)
        
        # Компактное поле ввода
        input_frame = ttk.LabelFrame(main_frame, text="⌨️ Ввод", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pangram_label = ttk.Label(
            input_frame,
            text=f'Введите: "{PANGRAM}"',
            font=(FONT_FAMILY, FONT_SIZE, 'bold'),
            foreground='darkblue'
        )
        self.pangram_label.pack(pady=(0, 5))
        
        # Прогресс ввода
        self.typing_progress_label = ttk.Label(
            input_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE-1, 'italic'),
            foreground='gray'
        )
        self.typing_progress_label.pack()
        
        self.text_entry = ttk.Entry(
            input_frame,
            width=60,
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.text_entry.pack(pady=5)
        
        self.status_label = ttk.Label(
            input_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE-1)
        )
        self.status_label.pack()
        
        # Компактное обучение модели
        training_frame = ttk.LabelFrame(main_frame, text="🤖 Обучение модели", padding=10)
        training_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.training_status = ttk.Label(
            training_frame,
            text="Статус: Ожидание сбора данных",
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.training_status.pack()
        
        self.training_progress = ttk.Progressbar(
            training_frame,
            length=500,
            mode='indeterminate'
        )
        self.training_progress.pack(pady=5)
        
        # Компактные кнопки в одну строку
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)
        
        self.submit_btn = ttk.Button(
            button_frame,
            text="💾 Сохранить образец",
            command=self.submit_sample,
            state=tk.DISABLED
        )
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(
            button_frame,
            text="🚀 Обучить модель",
            command=self.start_training,
            state=tk.DISABLED
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="❌ Отмена",
            command=self.window.destroy
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
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
                text="⏸️ Запись остановлена",
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
        current_text = self.text_entry.get()
        normalized_current = self._normalize_text(current_text)
        
        # Проверяем правильность префикса
        if len(normalized_current) > len(self.normalized_target):
            self._reset_input("❌ Текст слишком длинный")
            return
        
        is_correct_prefix = True
        for i, char in enumerate(normalized_current):
            if i >= len(self.normalized_target) or char != self.normalized_target[i]:
                is_correct_prefix = False
                break
        
        if not is_correct_prefix:
            self._reset_input("❌ Ошибка в тексте")
            return
        
        # Обновляем прогресс ввода
        progress_text = f"📝 {len(normalized_current)}/{len(self.normalized_target)} символов"
        if len(normalized_current) > 0:
            progress_text += f" | '{current_text[-min(8, len(current_text)):]}'"
        self.typing_progress_label.config(text=progress_text)
        
        # Проверяем завершенность
        if normalized_current == self.normalized_target:
            self.submit_btn.config(state=tk.NORMAL)
            self.status_label.config(
                text="✅ Текст введен правильно! Можно сохранить.",
                foreground="green"
            )
        else:
            self.submit_btn.config(state=tk.DISABLED)
            if len(normalized_current) > 0:
                self.status_label.config(
                    text="⌨️ Продолжайте ввод...",
                    foreground="blue"
                )
            else:
                self.status_label.config(
                    text="💭 Начните ввод панграммы",
                    foreground="black"
                )
    
    def _reset_input(self, message: str):
        """Сброс ввода при ошибке"""
        if self.is_recording:
            self.stop_recording()
            if self.session_id:
                if self.session_id in self.keystroke_auth.current_session:
                    del self.keystroke_auth.current_session[self.session_id]
                self.session_id = None
        
        self.text_entry.delete(0, tk.END)
        self.status_label.config(text=message, foreground="red")
        self.typing_progress_label.config(text="")
        self.submit_btn.config(state=tk.DISABLED)
        
        self.window.after(1500, self._clear_error_and_restart)
    
    def _clear_error_and_restart(self):
        """Очистка ошибки и подготовка к новому вводу"""
        self.status_label.config(text="🔄 Начните ввод заново", foreground="black")
        self.text_entry.focus()
    
    def submit_sample(self):
        """Сохранение образца"""
        current_text = self.text_entry.get()
        normalized_current = self._normalize_text(current_text)
        
        if normalized_current != self.normalized_target:
            messagebox.showwarning("❌ Предупреждение", "Введите панграмму полностью и правильно")
            return
        
        if self.session_id and self.is_recording:
            try:
                self.stop_recording()
                
                features = self.keystroke_auth.finish_recording(self.session_id, is_training=True)
                
                if not features or all(v == 0 for v in features.values()):
                    messagebox.showwarning(
                        "⚠️ Предупреждение", 
                        "Не удалось записать динамику нажатий.\nПопробуйте печатать медленнее."
                    )
                    self.text_entry.delete(0, tk.END)
                    self.text_entry.focus()
                    return
                
                self.current_sample += 1
                
                self.status_label.config(
                    text=f"✅ Образец {self.current_sample} сохранен",
                    foreground="green"
                )
                
                self.text_entry.delete(0, tk.END)
                self.typing_progress_label.config(text="")
                
                self.update_progress()
                
                # Пауза перед следующим образцом
                self.text_entry.config(state=tk.DISABLED)
                self.status_label.config(
                    text="⏳ Пауза... Готовьтесь к следующему образцу",
                    foreground="blue"
                )
                
                self.window.after(1500, self._enable_next_input)
                
            except Exception as e:
                messagebox.showerror("❌ Ошибка", f"Ошибка сохранения: {str(e)}")
                self.text_entry.delete(0, tk.END)
                self.text_entry.focus()
        else:
            messagebox.showwarning("⚠️ Предупреждение", "Нет активной записи")
    
    def _enable_next_input(self):
        """Разрешение ввода следующего образца"""
        self.text_entry.config(state=tk.NORMAL)
        self.text_entry.focus()
        self.status_label.config(text="📝 Готов к следующему образцу", foreground="black")
    
    def update_progress(self):
        """Обновление прогресса обучения"""
        progress = self.keystroke_auth.get_training_progress(self.user)
        
        self.current_sample = progress['current_samples']
        self.progress_label.config(
            text=f"📊 Образцов: {progress['current_samples']}/{progress['required_samples']}"
        )
        
        self.progress_bar['value'] = progress['current_samples']
        
        # Проверка готовности к обучению
        if progress['is_ready']:
            self.train_btn.config(state=tk.NORMAL)
            self.pangram_label.config(
                text="✅ Достаточно образцов! Можете обучить модель.",
                foreground="green"
            )
            self.training_status.config(
                text="✅ Статус: Готов к обучению модели"
            )
        else:
            remaining = progress['required_samples'] - progress['current_samples']
            self.pangram_label.config(
                text=f'Введите: "{PANGRAM}" (осталось {remaining} раз)',
                foreground="darkblue"
            )
    
    def start_training(self):
        """Запуск обучения модели"""
        if self.training_in_progress:
            return
        
        method_text = "продвинутое обучение с валидацией" if self.use_enhanced_training.get() else "базовое обучение"
        
        if messagebox.askyesno(
            "🚀 Подтверждение",
            f"Начать {method_text}?\n\nЭто может занять от нескольких секунд до нескольких минут."
        ):
            self.training_in_progress = True
            self.train_btn.config(state=tk.DISABLED, text="🔄 Обучение...")
            self.training_status.config(text="🤖 Статус: Обучение модели...")
            self.training_progress.start()
            
            # Запускаем обучение в отдельном потоке
            threading.Thread(target=self._train_model_thread, daemon=True).start()
    
    def _train_model_thread(self):
        """Обучение модели в отдельном потоке"""
        try:
            from ml.model_manager import ModelManager
            
            model_manager = ModelManager()
            
            # Выбираем метод обучения
            use_enhanced = self.use_enhanced_training.get()
            
            success, accuracy, message = model_manager.train_user_model(
                self.user.id, 
                use_enhanced_training=use_enhanced
            )
            
            # Обновляем интерфейс в главном потоке
            self.window.after(0, lambda: self._training_completed(success, accuracy, message, use_enhanced))
            
        except Exception as e:
            error_message = f"Ошибка при обучении: {str(e)}"
            self.window.after(0, lambda: self._training_completed(False, 0.0, error_message, False))
    
    def _training_completed(self, success: bool, accuracy: float, message: str, use_enhanced: bool):
        """Завершение обучения"""
        self.training_in_progress = False
        self.training_progress.stop()
        self.train_btn.config(state=tk.NORMAL, text="🚀 Обучить модель")
        
        if success:
            self.training_status.config(
                text=f"✅ Обучение завершено! Точность: {accuracy:.1%}"
            )
            
            # Дополнительная информация для продвинутого обучения
            if use_enhanced:
                try:
                    from ml.model_manager import ModelManager
                    model_manager = ModelManager()
                    report = model_manager.get_training_report(self.user.id)
                    
                    if report:
                        additional_info = f"""🔬 РЕЗУЛЬТАТЫ ПРОДВИНУТОГО ОБУЧЕНИЯ:

📊 Основные метрики:
• Точность на тесте: {accuracy:.1%}
• Время обучения: {report.get('training_duration', 0):.1f} сек
• Размер датасета: {report.get('dataset_size', 0)} образцов

⚙️ Оптимальные параметры:
{self._format_params(report.get('best_params', {}))}

📈 Кросс-валидация и анализ переобучения выполнены.
📄 Детальный отчет сохранен."""
                        
                        messagebox.showinfo("🎉 Продвинутое обучение завершено", additional_info)
                    else:
                        messagebox.showinfo("✅ Успех", f"{message}\n\nТеперь система готова к работе!")
                        
                except Exception as e:
                    print(f"Ошибка получения отчета: {e}")
                    messagebox.showinfo("✅ Успех", f"{message}\n\nТеперь система готова к работе!")
            else:
                messagebox.showinfo("✅ Успех", f"{message}\n\nТеперь система готова к работе!")
            
            self.on_complete()
            self.window.destroy()
        else:
            self.training_status.config(text="❌ Ошибка обучения")
            messagebox.showerror("❌ Ошибка", message)
    
    def _format_params(self, params: dict) -> str:
        """Форматирование параметров для отображения"""
        if not params:
            return "• Не определены"
        
        formatted = []
        for key, value in params.items():
            if key == 'n_neighbors':
                formatted.append(f"• Количество соседей: {value}")
            elif key == 'weights':
                formatted.append(f"• Веса: {value}")
            elif key == 'metric':
                formatted.append(f"• Метрика: {value}")
            elif key == 'algorithm':
                formatted.append(f"• Алгоритм: {value}")
            else:
                formatted.append(f"• {key}: {value}")
        
        return "\n".join(formatted)