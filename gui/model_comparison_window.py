# gui/model_comparison_window.py - Окно сравнения KNN и Random Forest

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import List, Dict, Tuple
import time

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from models.user import User
from ml.knn_classifier import KNNAuthenticator
from ml.random_forest_classifier import RandomForestAuthenticator
from ml.feature_extractor import FeatureExtractor
from utils.database import DatabaseManager
from config import FONT_FAMILY, FONT_SIZE, THRESHOLD_ACCURACY


class ModelComparisonWindow:
    """Окно для сравнения KNN и Random Forest моделей"""
    
    def __init__(self, parent, user: User):
        self.parent = parent
        self.user = user
        self.db = DatabaseManager()
        self.feature_extractor = FeatureExtractor()
        
        # Модели
        self.knn_model = None
        self.rf_model = None
        self.comparison_results = {}
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Сравнение моделей - {user.username}")
        self.window.geometry("1000x700")
        self.window.resizable(True, True)
        self.window.minsize(900, 600)
        
        self.create_widgets()
        
        # Центрирование окна
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Автоматический запуск сравнения
        self.window.after(100, self.run_comparison)
    
    def create_widgets(self):
        """Создание интерфейса"""
        # Заголовок
        title_frame = ttk.Frame(self.window)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(
            title_frame,
            text="Сравнение методов машинного обучения",
            font=(FONT_FAMILY, 16, 'bold')
        ).pack()
        
        ttk.Label(
            title_frame,
            text="K-Nearest Neighbors (KNN) vs Random Forest",
            font=(FONT_FAMILY, 12)
        ).pack()
        
        # Прогресс
        self.progress_frame = ttk.Frame(self.window)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.progress_label = ttk.Label(
            self.progress_frame,
            text="Подготовка к сравнению...",
            font=(FONT_FAMILY, 11)
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(pady=5)
        
        # Notebook для результатов
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Вкладки
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="Общее сравнение")
        
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="Детальные метрики")
        
        self.features_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.features_tab, text="Важность признаков")
        
        self.recommendation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendation_tab, text="Рекомендации")
    
    def run_comparison(self):
        """Запуск сравнения моделей"""
        self.progress_bar.start()
        
        try:
            # Получение данных
            self.progress_label.config(text="Загрузка обучающих данных...")
            self.window.update()
            
            samples = self.db.get_user_training_samples(self.user.id)
            if len(samples) < 10:
                messagebox.showerror(
                    "Ошибка",
                    "Недостаточно данных для сравнения.\nНеобходимо минимум 10 образцов."
                )
                self.window.destroy()
                return
            
            # Извлечение признаков
            features = []
            for sample in samples:
                features.append([
                    sample.features.get('avg_dwell_time', 0),
                    sample.features.get('std_dwell_time', 0),
                    sample.features.get('avg_flight_time', 0),
                    sample.features.get('std_flight_time', 0),
                    sample.features.get('typing_speed', 0),
                    sample.features.get('total_typing_time', 0)
                ])
            
            # Обучение KNN
            self.progress_label.config(text="Обучение KNN модели...")
            self.window.update()
            
            self.knn_model = KNNAuthenticator()
            knn_start = time.time()
            knn_accuracy, knn_std = self.knn_model.train(features)
            knn_time = time.time() - knn_start
            
            # Обучение Random Forest  
            self.progress_label.config(text="Обучение Random Forest модели...")
            self.window.update()
            
            self.rf_model = RandomForestAuthenticator()
            rf_start = time.time()
            rf_accuracy, rf_std = self.rf_model.train(features, optimize_params=True)
            rf_time = time.time() - rf_start
            
            # Кросс-валидация для получения метрик
            self.progress_label.config(text="Вычисление метрик производительности...")
            self.window.update()
            
            knn_metrics = self.evaluate_model(self.knn_model, features)
            rf_metrics = self.evaluate_model(self.rf_model, features)
            
            # Сохранение результатов
            self.comparison_results = {
                'knn': {
                    'accuracy': knn_accuracy,
                    'std': knn_std,
                    'training_time': knn_time,
                    'metrics': knn_metrics,
                    'params': {'n_neighbors': self.knn_model.n_neighbors}
                },
                'rf': {
                    'accuracy': rf_accuracy,
                    'std': rf_std,
                    'training_time': rf_time,
                    'metrics': rf_metrics,
                    'params': self.rf_model.best_params or {
                        'n_estimators': self.rf_model.n_estimators,
                        'max_depth': self.rf_model.max_depth
                    },
                    'feature_importance': self.rf_model.get_feature_importance()
                }
            }
            
            # Отображение результатов
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            self.display_results()
            
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Ошибка", f"Ошибка при сравнении: {str(e)}")
            self.window.destroy()
    
    def evaluate_model(self, model, features: List[List[float]]) -> Dict:
        """Оценка модели через симуляцию"""
        # Разделение на обучающую и тестовую выборки
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        test_features = features[split_idx:]
        
        # Переобучение на обучающей выборке
        model.train(train_features)
        
        # Тестирование
        tp = 0  # True Positive
        fn = 0  # False Negative
        
        # Тест на легитимных образцах
        for feature in test_features:
            success, confidence = model.authenticate(feature, THRESHOLD_ACCURACY)
            if success:
                tp += 1
            else:
                fn += 1
        
        # Симуляция атак (искажение признаков)
        fp = 0  # False Positive
        tn = 0  # True Negative
        
        for _ in range(len(test_features)):
            # Создание "поддельного" образца с отклонением
            fake_feature = []
            for feat in features[0]:
                # Добавляем случайное отклонение 20-50%
                deviation = np.random.uniform(0.2, 0.5)
                fake_value = feat * (1 + np.random.choice([-1, 1]) * deviation)
                fake_feature.append(fake_value)
            
            success, confidence = model.authenticate(fake_feature, THRESHOLD_ACCURACY)
            if success:
                fp += 1
            else:
                tn += 1
        
        # Расчет метрик
        total_legitimate = tp + fn
        total_impostor = fp + tn
        
        far = fp / total_impostor if total_impostor > 0 else 0
        frr = fn / total_legitimate if total_legitimate > 0 else 0
        accuracy = (tp + tn) / (total_legitimate + total_impostor)
        
        return {
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'far': far, 'frr': frr,
            'accuracy': accuracy,
            'eer': (far + frr) / 2  # Приближенный EER
        }
    
    def display_results(self):
        """Отображение результатов сравнения"""
        self.display_overview()
        self.display_metrics()
        self.display_features()
        self.display_recommendations()
    
    def display_overview(self):
        """Общее сравнение"""
        frame = ttk.Frame(self.overview_tab, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        ttk.Label(
            frame,
            text="Результаты сравнения моделей",
            font=(FONT_FAMILY, 14, 'bold')
        ).pack(pady=10)
        
        # Таблица сравнения
        columns = ('Параметр', 'KNN', 'Random Forest', 'Лучше')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200)
        
        # Данные для сравнения
        knn = self.comparison_results['knn']
        rf = self.comparison_results['rf']
        
        comparisons = [
            ('Точность (CV)', f"{knn['accuracy']:.2%}", f"{rf['accuracy']:.2%}", 
             'Random Forest' if rf['accuracy'] > knn['accuracy'] else 'KNN'),
            
            ('Стандартное отклонение', f"{knn['std']:.4f}", f"{rf['std']:.4f}",
             'Random Forest' if rf['std'] < knn['std'] else 'KNN'),
            
            ('Время обучения', f"{knn['training_time']:.3f}с", f"{rf['training_time']:.3f}с",
             'KNN' if knn['training_time'] < rf['training_time'] else 'Random Forest'),
            
            ('FAR', f"{knn['metrics']['far']:.2%}", f"{rf['metrics']['far']:.2%}",
             'Random Forest' if rf['metrics']['far'] < knn['metrics']['far'] else 'KNN'),
            
            ('FRR', f"{knn['metrics']['frr']:.2%}", f"{rf['metrics']['frr']:.2%}",
             'Random Forest' if rf['metrics']['frr'] < knn['metrics']['frr'] else 'KNN'),
            
            ('EER', f"{knn['metrics']['eer']:.2%}", f"{rf['metrics']['eer']:.2%}",
             'Random Forest' if rf['metrics']['eer'] < knn['metrics']['eer'] else 'KNN'),
        ]
        
        for comp in comparisons:
            tree.insert('', 'end', values=comp)
        
        tree.pack(pady=10)
        
        # График сравнения
        if MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            
            metrics = ['Точность', 'FAR', 'FRR', 'EER']
            knn_values = [
                knn['accuracy'] * 100,
                knn['metrics']['far'] * 100,
                knn['metrics']['frr'] * 100,
                knn['metrics']['eer'] * 100
            ]
            rf_values = [
                rf['accuracy'] * 100,
                rf['metrics']['far'] * 100,
                rf['metrics']['frr'] * 100,
                rf['metrics']['eer'] * 100
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, knn_values, width, label='KNN', alpha=0.8)
            ax.bar(x + width/2, rf_values, width, label='Random Forest', alpha=0.8)
            
            ax.set_ylabel('Процент (%)')
            ax.set_title('Сравнение метрик')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_metrics(self):
        """Детальные метрики"""
        frame = ttk.Frame(self.metrics_tab, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # KNN метрики
        knn_frame = ttk.LabelFrame(frame, text="K-Nearest Neighbors", padding=15)
        knn_frame.pack(fill=tk.X, pady=10)
        
        knn_text = self.format_metrics_text("KNN", self.comparison_results['knn'])
        ttk.Label(knn_frame, text=knn_text, font=(FONT_FAMILY, 10)).pack(anchor=tk.W)
        
        # Random Forest метрики
        rf_frame = ttk.LabelFrame(frame, text="Random Forest", padding=15)
        rf_frame.pack(fill=tk.X, pady=10)
        
        rf_text = self.format_metrics_text("Random Forest", self.comparison_results['rf'])
        ttk.Label(rf_frame, text=rf_text, font=(FONT_FAMILY, 10)).pack(anchor=tk.W)
    
    def display_features(self):
        """Важность признаков (только для Random Forest)"""
        frame = ttk.Frame(self.features_tab, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            frame,
            text="Важность признаков в Random Forest",
            font=(FONT_FAMILY, 14, 'bold')
        ).pack(pady=10)
        
        importance = self.comparison_results['rf'].get('feature_importance', {})
        
        if importance and MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            features = list(importance.keys())
            values = list(importance.values())
            
            # Сортировка по важности
            sorted_idx = np.argsort(values)[::-1]
            features = [features[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            # Русские названия
            feature_names_ru = {
                'avg_dwell_time': 'Среднее время удержания',
                'std_dwell_time': 'СКО времени удержания',
                'avg_flight_time': 'Среднее время между клавишами',
                'std_flight_time': 'СКО времени между клавишами',
                'typing_speed': 'Скорость печати',
                'total_typing_time': 'Общее время набора'
            }
            
            features_ru = [feature_names_ru.get(f, f) for f in features]
            
            bars = ax.barh(features_ru, values)
            ax.set_xlabel('Важность')
            ax.set_title('Какие признаки наиболее важны для идентификации')
            
            # Цветовая градация
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Добавление значений
            for i, (feature, value) in enumerate(zip(features_ru, values)):
                ax.text(value + 0.01, i, f'{value:.3f}', va='center')
            
            ax.set_xlim(0, max(values) * 1.15)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(
                frame,
                text="Важность признаков доступна только для Random Forest",
                font=(FONT_FAMILY, 12)
            ).pack(pady=50)
    
    def display_recommendations(self):
        """Рекомендации по выбору модели"""
        frame = ttk.Frame(self.recommendation_tab, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            frame,
            text="Рекомендации по выбору модели",
            font=(FONT_FAMILY, 14, 'bold')
        ).pack(pady=10)
        
        # Анализ результатов
        knn = self.comparison_results['knn']
        rf = self.comparison_results['rf']
        
        # Подсчет победителя
        knn_wins = 0
        rf_wins = 0
        
        if knn['accuracy'] > rf['accuracy']:
            knn_wins += 1
        else:
            rf_wins += 1
            
        if knn['metrics']['far'] < rf['metrics']['far']:
            knn_wins += 1
        else:
            rf_wins += 1
            
        if knn['metrics']['frr'] < rf['metrics']['frr']:
            knn_wins += 1
        else:
            rf_wins += 1
            
        if knn['training_time'] < rf['training_time']:
            knn_wins += 1
        else:
            rf_wins += 1
        
        # Генерация рекомендаций
        recommendations = []
        
        # Общая рекомендация
        if rf_wins > knn_wins:
            winner = "Random Forest"
            winner_color = "darkgreen"
        else:
            winner = "K-Nearest Neighbors"
            winner_color = "darkblue"
        
        recommendations.append(f"🏆 Рекомендуемая модель: {winner}")
        recommendations.append("")
        
        # Детальный анализ
        recommendations.append("📊 Анализ по критериям:")
        recommendations.append("")
        
        # Точность
        if rf['accuracy'] > knn['accuracy']:
            recommendations.append(f"✓ Random Forest показывает лучшую точность ({rf['accuracy']:.2%} vs {knn['accuracy']:.2%})")
        else:
            recommendations.append(f"✓ KNN показывает лучшую точность ({knn['accuracy']:.2%} vs {rf['accuracy']:.2%})")
        
        # Безопасность
        recommendations.append("")
        recommendations.append("🔒 Безопасность:")
        
        if rf['metrics']['far'] < knn['metrics']['far']:
            recommendations.append(f"✓ Random Forest имеет меньший FAR (безопаснее от взлома): {rf['metrics']['far']:.2%}")
        else:
            recommendations.append(f"✓ KNN имеет меньший FAR (безопаснее от взлома): {knn['metrics']['far']:.2%}")
        
        if rf['metrics']['frr'] < knn['metrics']['frr']:
            recommendations.append(f"✓ Random Forest имеет меньший FRR (удобнее для пользователя): {rf['metrics']['frr']:.2%}")
        else:
            recommendations.append(f"✓ KNN имеет меньший FRR (удобнее для пользователя): {knn['metrics']['frr']:.2%}")
        
        # Производительность
        recommendations.append("")
        recommendations.append("⚡ Производительность:")
        
        if knn['training_time'] < rf['training_time']:
            recommendations.append(f"✓ KNN обучается быстрее ({knn['training_time']:.3f}с vs {rf['training_time']:.3f}с)")
        else:
            recommendations.append(f"✓ Random Forest обучается быстрее ({rf['training_time']:.3f}с vs {knn['training_time']:.3f}с)")
        
        # Особенности моделей
        recommendations.append("")
        recommendations.append("📝 Особенности моделей:")
        recommendations.append("")
        
        recommendations.append("K-Nearest Neighbors:")
        recommendations.append("• Простой и интерпретируемый алгоритм")
        recommendations.append("• Хорошо работает с малым количеством данных")
        recommendations.append("• Быстрое обучение")
        recommendations.append("• Чувствителен к выбросам")
        
        recommendations.append("")
        recommendations.append("Random Forest:")
        recommendations.append("• Более устойчив к шуму и выбросам")
        recommendations.append("• Показывает важность признаков")
        recommendations.append("• Лучше обобщает на новых данных")
        recommendations.append("• Требует больше времени на обучение")
        
        # Финальные рекомендации
        recommendations.append("")
        recommendations.append("💡 Рекомендации для вашего случая:")
        recommendations.append("")
        
        if len(self.db.get_user_training_samples(self.user.id)) < 30:
            recommendations.append("• У вас мало данных (<30 образцов), KNN может работать лучше")
        else:
            recommendations.append("• У вас достаточно данных, Random Forest может дать лучшие результаты")
        
        if rf['metrics']['far'] < 0.05 and rf['metrics']['frr'] < 0.05:
            recommendations.append("• Random Forest показывает отличные метрики безопасности")
        
        if knn['training_time'] < 0.1:
            recommendations.append("• Если важна скорость обучения, выбирайте KNN")
        
        # Отображение рекомендаций
        text_widget = tk.Text(frame, wrap=tk.WORD, font=(FONT_FAMILY, 11), height=25)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, rec in enumerate(recommendations):
            if rec.startswith("🏆"):
                text_widget.insert(tk.END, rec + "\n", "winner")
                text_widget.tag_config("winner", font=(FONT_FAMILY, 13, 'bold'), foreground=winner_color)
            elif rec.startswith(("📊", "🔒", "⚡", "📝", "💡")):
                text_widget.insert(tk.END, rec + "\n", "header")
                text_widget.tag_config("header", font=(FONT_FAMILY, 12, 'bold'))
            elif rec.startswith("✓"):
                text_widget.insert(tk.END, rec + "\n", "positive")
                text_widget.tag_config("positive", foreground="darkgreen")
            else:
                text_widget.insert(tk.END, rec + "\n")
        
        text_widget.config(state=tk.DISABLED)
    
    def format_metrics_text(self, model_name: str, results: Dict) -> str:
        """Форматирование текста метрик"""
        metrics = results['metrics']
        params = results['params']
        
        text = f"""Параметры модели:
"""
        
        if model_name == "KNN":
            text += f"• Количество соседей (K): {params.get('n_neighbors', 5)}\n"
            text += f"• Метрика расстояния: Евклидова\n"
        else:
            text += f"• Количество деревьев: {params.get('n_estimators', 100)}\n"
            text += f"• Максимальная глубина: {params.get('max_depth', 'Не ограничена')}\n"
            if 'min_samples_split' in params:
                text += f"• Мин. образцов для разделения: {params['min_samples_split']}\n"
            if 'min_samples_leaf' in params:
                text += f"• Мин. образцов в листе: {params['min_samples_leaf']}\n"
        
        text += f"""
Результаты обучения:
• Точность (cross-validation): {results['accuracy']:.2%}
• Стандартное отклонение: {results['std']:.4f}
• Время обучения: {results['training_time']:.3f} секунд

Матрица ошибок:
• True Positive (TP): {metrics['tp']} - Правильно принятые
• False Negative (FN): {metrics['fn']} - Ошибочно отклоненные
• False Positive (FP): {metrics['fp']} - Ошибочно принятые
• True Negative (TN): {metrics['tn']} - Правильно отклоненные

Метрики безопасности:
• FAR (False Acceptance Rate): {metrics['far']:.2%}
• FRR (False Rejection Rate): {metrics['frr']:.2%}
• EER (Equal Error Rate): {metrics['eer']:.2%}
• Общая точность: {metrics['accuracy']:.2%}
"""
        
        return text