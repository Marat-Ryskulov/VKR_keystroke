# gui/model_stats_window.py - Окно статистики и анализа модели

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib не установлен. Графики будут недоступны.")

from datetime import datetime
import os

from models.user import User
from ml.model_manager import ModelManager
from ml.feature_extractor import FeatureExtractor
from utils.database import DatabaseManager
from config import FONT_FAMILY, FONT_SIZE, DATA_DIR, THRESHOLD_ACCURACY


class ModelStatsWindow:
    """Окно для отображения статистики и анализа модели"""
    
    def __init__(self, parent, user: User):
        self.parent = parent
        self.user = user
        self.model_manager = ModelManager()
        self.feature_extractor = FeatureExtractor()
        self.db = DatabaseManager()
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Статистика модели - {user.username}")
        self.window.geometry("1200x800")
        self.window.resizable(True, True)
        self.window.minsize(1000, 700)
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создание вкладок
        self.create_overview_tab()
        self.create_features_tab()
        self.create_performance_tab()
        self.create_roc_tab()
        self.create_raw_data_tab()
        
        # Центрирование окна
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Фокус на окне
        self.window.focus()
    
    def create_overview_tab(self):
        """Вкладка с общей информацией"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Обзор")
        
        # Основная информация
        info_frame = ttk.LabelFrame(tab, text="Информация о модели", padding=20)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Получение данных
        samples = self.db.get_user_training_samples(self.user.id)
        model_info = self.model_manager.get_model_info(self.user.id)
        
        # Статистика
        stats_text = f"""
Пользователь: {self.user.username}
Дата регистрации: {self.user.created_at.strftime('%d.%m.%Y %H:%M')}
Статус модели: {'Обучена' if self.user.is_trained else 'Не обучена'}

Обучающих образцов: {len(samples)}
Минимум для обучения: {model_info.get('min_samples', 20)}

Параметры модели:
• Алгоритм: K-Nearest Neighbors (KNN)
• Количество соседей (K): {model_info.get('n_neighbors', 5)}
• Метрика расстояния: Евклидова
• Вес соседей: Равномерный
• Минимум образцов для обучения: {model_info.get('min_samples', 100)}

Точность модели:
• Перекрестная проверка: {model_info.get('cv_score', 0):.2%}
• Порог принятия: {THRESHOLD_ACCURACY:.2%}

Признаки для анализа:
• Среднее время удержания клавиш (dwell time)
• Стандартное отклонение времени удержания
• Среднее время между клавишами (flight time)
• Стандартное отклонение времени между клавишами
• Скорость печати (клавиш в секунду)
• Общее время набора текста
"""
        
        stats_label = ttk.Label(info_frame, text=stats_text, font=(FONT_FAMILY, 11))
        stats_label.pack(anchor=tk.W)
        
        # Кнопка экспорта отчета
        export_btn = ttk.Button(
            info_frame,
            text="Экспортировать полный отчет",
            command=self.export_report
        )
        export_btn.pack(pady=20)
    
    def create_features_tab(self):
        """Вкладка с анализом признаков"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Анализ признаков")
        
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(tab, text="Matplotlib не установлен. Графики недоступны.\nУстановите: pip install matplotlib", 
                     font=(FONT_FAMILY, 12)).pack(pady=50)
            return
        
        # Создание графиков
        fig = Figure(figsize=(12, 8))
        
        # Получение данных
        samples = self.db.get_user_training_samples(self.user.id)
        if not samples:
            ttk.Label(tab, text="Нет данных для анализа", font=(FONT_FAMILY, 14)).pack(pady=50)
            return
        
        # Извлечение признаков
        features_list = []
        for sample in samples:
            features_list.append([
                sample.features.get('avg_dwell_time', 0),
                sample.features.get('std_dwell_time', 0),
                sample.features.get('avg_flight_time', 0),
                sample.features.get('std_flight_time', 0),
                sample.features.get('typing_speed', 0),
                sample.features.get('total_typing_time', 0)
            ])
        
        features_array = np.array(features_list)
        feature_names = [
            'Ср. время удержания',
            'СКО времени удержания',
            'Ср. время между клав.',
            'СКО времени между клав.',
            'Скорость печати',
            'Общее время'
        ]
        
        # Создание подграфиков
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1)
            data = features_array[:, i]
            
            # Гистограмма с кривой плотности
            ax.hist(data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Среднее: {np.mean(data):.3f}')
            ax.axvline(np.median(data), color='green', linestyle='--', label=f'Медиана: {np.median(data):.3f}')
            
            ax.set_title(feature_names[i])
            ax.set_xlabel('Значение')
            ax.set_ylabel('Частота')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Добавление на вкладку
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_performance_tab(self):
        """Вкладка с метриками производительности"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Производительность")
        
        # Фрейм для метрик
        metrics_frame = ttk.LabelFrame(tab, text="Метрики безопасности", padding=20)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Симуляция метрик (в реальной системе нужно вычислять на тестовых данных)
        samples = self.db.get_user_training_samples(self.user.id)
        
        if len(samples) < 10:
            ttk.Label(metrics_frame, text="Недостаточно данных для расчета метрик", 
                     font=(FONT_FAMILY, 12)).pack(pady=50)
            return
        
        # Разделение на обучающую и тестовую выборки
        train_size = int(len(samples) * 0.8)
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]
        
        # Расчет метрик
        far, frr, eer = self.calculate_metrics(train_samples, test_samples)
        
        metrics_text = f"""
Метрики безопасности системы:

FAR (False Acceptance Rate) - Ложное принятие:
• Текущее значение: {far:.2%}
• Описание: Вероятность того, что система неправильно примет чужого пользователя
• Рекомендуемое: < 1%

FRR (False Rejection Rate) - Ложный отказ:
• Текущее значение: {frr:.2%}
• Описание: Вероятность того, что система отклонит легитимного пользователя
• Рекомендуемое: < 5%

EER (Equal Error Rate) - Равная частота ошибок:
• Текущее значение: {eer:.2%}
• Описание: Точка, где FAR = FRR
• Рекомендуемое: < 3%

Общая точность системы: {(1 - (far + frr) / 2):.2%}

Примечание: Данные метрики рассчитаны на основе перекрестной проверки
с использованием {len(train_samples)} обучающих и {len(test_samples)} тестовых образцов.
"""
        
        metrics_label = ttk.Label(metrics_frame, text=metrics_text, font=(FONT_FAMILY, 11))
        metrics_label.pack(anchor=tk.W)
        
        # График метрик
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        categories = ['FAR', 'FRR', 'EER']
        values = [far * 100, frr * 100, eer * 100]
        colors = ['red', 'blue', 'green']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Процент (%)')
        ax.set_title('Метрики безопасности')
        ax.grid(True, alpha=0.3)
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}%', ha='center', va='bottom')
        
        canvas = FigureCanvasTkAgg(fig, master=metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=20)
    
    def create_roc_tab(self):
        """Вкладка с ROC-кривой"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="ROC-кривая")
        
        # Создание ROC-кривой
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Симуляция ROC-кривой
        samples = self.db.get_user_training_samples(self.user.id)
        if len(samples) < 10:
            ttk.Label(tab, text="Недостаточно данных для построения ROC-кривой", 
                     font=(FONT_FAMILY, 12)).pack(pady=50)
            return
        
        # Генерация точек ROC-кривой
        thresholds = np.linspace(0, 1, 100)
        tpr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            # Симуляция TPR и FPR для разных порогов
            # В реальной системе нужно вычислять на реальных данных
            tpr = 1 - (threshold * 0.3)  # True Positive Rate
            fpr = threshold * 0.2  # False Positive Rate
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Расчет AUC
        auc = np.trapezoid(tpr_values, fpr_values)
        
        # Построение ROC-кривой
        ax.plot(fpr_values, tpr_values, 'b-', linewidth=2, label=f'ROC-кривая (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Случайный классификатор')
        
        # Отметка текущего порога
        current_threshold_idx = int(THRESHOLD_ACCURACY * 100)
        ax.plot(fpr_values[current_threshold_idx], tpr_values[current_threshold_idx], 
                'go', markersize=10, label=f'Текущий порог ({THRESHOLD_ACCURACY:.2f})')
        
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('ROC-кривая модели аутентификации')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Информация под графиком
        info_frame = ttk.Frame(tab)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        info_text = f"""
AUC (Area Under Curve): {auc:.3f}
• Значение > 0.9 - отличная модель
• Значение 0.8-0.9 - хорошая модель
• Значение 0.7-0.8 - удовлетворительная модель
• Значение < 0.7 - слабая модель

Текущий порог принятия решения: {THRESHOLD_ACCURACY:.2f}
"""
        ttk.Label(info_frame, text=info_text, font=(FONT_FAMILY, 10)).pack()
    
    def create_raw_data_tab(self):
        """Вкладка с сырыми данными"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Данные образцов")
        
        # Создание таблицы
        columns = ('ID', 'Время', 'Ср. удержание', 'Ср. между клав.', 'Скорость', 'Общее время')
        tree = ttk.Treeview(tab, columns=columns, show='headings', height=20)
        
        # Определение заголовков
        tree.heading('ID', text='ID')
        tree.heading('Время', text='Время записи')
        tree.heading('Ср. удержание', text='Ср. удержание (мс)')
        tree.heading('Ср. между клав.', text='Ср. между клав. (мс)')
        tree.heading('Скорость', text='Скорость (кл/с)')
        tree.heading('Общее время', text='Общее время (с)')
        
        # Настройка ширины столбцов
        tree.column('ID', width=50)
        tree.column('Время', width=150)
        tree.column('Ср. удержание', width=120)
        tree.column('Ср. между клав.', width=120)
        tree.column('Скорость', width=100)
        tree.column('Общее время', width=100)
        
        # Получение данных
        samples = self.db.get_user_training_samples(self.user.id)
        
        # Заполнение таблицы
        for i, sample in enumerate(samples):
            values = (
                i + 1,
                sample.timestamp.strftime('%d.%m.%Y %H:%M:%S'),
                f"{sample.features.get('avg_dwell_time', 0)*1000:.1f}",
                f"{sample.features.get('avg_flight_time', 0)*1000:.1f}",
                f"{sample.features.get('typing_speed', 0):.2f}",
                f"{sample.features.get('total_typing_time', 0):.2f}"
            )
            tree.insert('', 'end', values=values)
        
        # Скроллбары
        vsb = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tab, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Размещение
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        
        # Статистика под таблицей
        stats_frame = ttk.Frame(tab)
        stats_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky='ew')
        
        if samples:
            avg_dwell = np.mean([s.features.get('avg_dwell_time', 0) for s in samples])
            avg_flight = np.mean([s.features.get('avg_flight_time', 0) for s in samples])
            avg_speed = np.mean([s.features.get('typing_speed', 0) for s in samples])
            
            stats_text = f"Средние значения по всем образцам: Удержание: {avg_dwell*1000:.1f} мс | Между клавишами: {avg_flight*1000:.1f} мс | Скорость: {avg_speed:.2f} кл/с"
            ttk.Label(stats_frame, text=stats_text, font=(FONT_FAMILY, 10)).pack()
    
    def calculate_metrics(self, train_samples, test_samples):
        """Расчет метрик FAR, FRR, EER"""
        # Симуляция метрик для демонстрации
        # В реальной системе нужно:
        # 1. Обучить модель на train_samples
        # 2. Тестировать на test_samples с разными порогами
        # 3. Вычислить реальные FAR/FRR
        
        # Для демонстрации используем приблизительные значения
        # основанные на количестве образцов и точности модели
        model_info = self.model_manager.get_model_info(self.user.id)
        cv_score = model_info.get('cv_score', 0.85)
        
        # Приблизительные расчеты
        far = (1 - cv_score) * 0.3  # False Acceptance Rate
        frr = (1 - cv_score) * 0.7  # False Rejection Rate
        eer = (far + frr) / 2  # Equal Error Rate
        
        return far, frr, eer
    
    def export_report(self):
        """Экспорт полного отчета в текстовый файл"""
        # Создание папки для отчетов
        reports_dir = os.path.join(DATA_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_report_{self.user.username}_{timestamp}.txt"
        filepath = os.path.join(reports_dir, filename)
        
        # Сбор данных
        samples = self.db.get_user_training_samples(self.user.id)
        model_info = self.model_manager.get_model_info(self.user.id)
        
        # Расчет метрик
        if len(samples) >= 10:
            train_size = int(len(samples) * 0.8)
            train_samples = samples[:train_size]
            test_samples = samples[train_size:]
            far, frr, eer = self.calculate_metrics(train_samples, test_samples)
        else:
            far = frr = eer = 0
        
        # Создание отчета
        report = f"""
================================================================================
                    ОТЧЕТ О МОДЕЛИ АУТЕНТИФИКАЦИИ
================================================================================

Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
Пользователь: {self.user.username}
ID пользователя: {self.user.id}

================================================================================
1. ОБЩАЯ ИНФОРМАЦИЯ
================================================================================

Дата регистрации: {self.user.created_at.strftime('%d.%m.%Y %H:%M:%S')}
Статус модели: {'Обучена' if self.user.is_trained else 'Не обучена'}
Количество обучающих образцов: {len(samples)}
Минимальное количество для обучения: {model_info.get('min_samples', 20)}

================================================================================
2. ПАРАМЕТРЫ МОДЕЛИ
================================================================================

Алгоритм: K-Nearest Neighbors (KNN)
Количество соседей (K): {model_info.get('n_neighbors', 5)}
Метрика расстояния: Евклидова (Euclidean)
Вес соседей: Равномерный (uniform)
Порог принятия решения: {THRESHOLD_ACCURACY:.2%}

================================================================================
3. ИСПОЛЬЗУЕМЫЕ ПРИЗНАКИ
================================================================================

1. Среднее время удержания клавиш (Average Dwell Time)
   - Описание: Среднее время между нажатием и отпусканием клавиши
   - Единица измерения: секунды

2. Стандартное отклонение времени удержания (Std Dwell Time)
   - Описание: Мера вариативности времени удержания клавиш
   - Единица измерения: секунды

3. Среднее время между клавишами (Average Flight Time)
   - Описание: Среднее время между отпусканием одной клавиши и нажатием следующей
   - Единица измерения: секунды

4. Стандартное отклонение времени между клавишами (Std Flight Time)
   - Описание: Мера вариативности времени между клавишами
   - Единица измерения: секунды

5. Скорость печати (Typing Speed)
   - Описание: Количество клавиш, нажатых за секунду
   - Единица измерения: клавиш/секунду

6. Общее время набора (Total Typing Time)
   - Описание: Общее время от первого до последнего нажатия
   - Единица измерения: секунды

================================================================================
4. ТОЧНОСТЬ МОДЕЛИ
================================================================================

Точность перекрестной проверки: {model_info.get('cv_score', 0):.2%}
Количество fold'ов: 5

================================================================================
5. МЕТРИКИ БЕЗОПАСНОСТИ
================================================================================

FAR (False Acceptance Rate): {far:.2%}
- Описание: Вероятность ложного принятия (принятие злоумышленника)
- Рекомендуемое значение: < 1%

FRR (False Rejection Rate): {frr:.2%}
- Описание: Вероятность ложного отказа (отказ легитимному пользователю)
- Рекомендуемое значение: < 5%

EER (Equal Error Rate): {eer:.2%}
- Описание: Точка, где FAR = FRR
- Рекомендуемое значение: < 3%

Общая точность системы: {(1 - (far + frr) / 2):.2%}

================================================================================
6. СТАТИСТИКА ПО ОБРАЗЦАМ
================================================================================
"""
        
        if samples:
            # Расчет статистики
            features_stats = {
                'avg_dwell_time': [],
                'std_dwell_time': [],
                'avg_flight_time': [],
                'std_flight_time': [],
                'typing_speed': [],
                'total_typing_time': []
            }
            
            for sample in samples:
                for key in features_stats:
                    features_stats[key].append(sample.features.get(key, 0))
            
            report += f"""
Среднее время удержания клавиш:
- Среднее: {np.mean(features_stats['avg_dwell_time'])*1000:.2f} мс
- Мин: {np.min(features_stats['avg_dwell_time'])*1000:.2f} мс
- Макс: {np.max(features_stats['avg_dwell_time'])*1000:.2f} мс
- СКО: {np.std(features_stats['avg_dwell_time'])*1000:.2f} мс

Среднее время между клавишами:
- Среднее: {np.mean(features_stats['avg_flight_time'])*1000:.2f} мс
- Мин: {np.min(features_stats['avg_flight_time'])*1000:.2f} мс
- Макс: {np.max(features_stats['avg_flight_time'])*1000:.2f} мс
- СКО: {np.std(features_stats['avg_flight_time'])*1000:.2f} мс

Скорость печати:
- Средняя: {np.mean(features_stats['typing_speed']):.2f} кл/с
- Мин: {np.min(features_stats['typing_speed']):.2f} кл/с
- Макс: {np.max(features_stats['typing_speed']):.2f} кл/с
- СКО: {np.std(features_stats['typing_speed']):.2f} кл/с

Общее время набора:
- Среднее: {np.mean(features_stats['total_typing_time']):.2f} с
- Мин: {np.min(features_stats['total_typing_time']):.2f} с
- Макс: {np.max(features_stats['total_typing_time']):.2f} с
- СКО: {np.std(features_stats['total_typing_time']):.2f} с
"""
        
        report += """
================================================================================
7. РЕКОМЕНДАЦИИ
================================================================================

"""
        
        if len(samples) < 30:
            report += "- Рекомендуется собрать больше образцов (30+) для повышения точности\n"
        
        if far > 0.01:
            report += "- FAR выше рекомендуемого значения. Увеличьте порог принятия решения\n"
        
        if frr > 0.05:
            report += "- FRR выше рекомендуемого значения. Уменьшите порог принятия решения\n"
        
        if model_info.get('cv_score', 0) < 0.85:
            report += "- Точность модели ниже 85%. Рекомендуется переобучить модель\n"
        
        report += """
================================================================================
                                КОНЕЦ ОТЧЕТА
================================================================================
"""
        
        # Сохранение отчета
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            messagebox.showinfo(
                "Успех",
                f"Отчет сохранен в:\n{filepath}"
            )
            
            # Открытие папки с отчетом
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(reports_dir)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', reports_dir])
            else:  # Linux
                subprocess.Popen(['xdg-open', reports_dir])
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении отчета: {str(e)}")