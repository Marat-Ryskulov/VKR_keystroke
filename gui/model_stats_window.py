# gui/model_stats_window.py - Исправленная версия с правильными импортами

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# Исправленный импорт для совместимости с разными версиями matplotlib
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
except ImportError:
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTk as FigureCanvas
    except ImportError:
        # Fallback для очень старых версий
        from matplotlib.backends.backend_tkagg import FigureCanvasTkinter as FigureCanvas

import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from ml.model_manager import ModelManager
from utils.database import DatabaseManager
from config import FONT_FAMILY

# Настройка matplotlib для работы с tkinter
plt.style.use('default')

class ModelStatsWindow:
    """Окно статистики модели с реальными данными"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        self.model_manager = ModelManager()
        self.db = DatabaseManager()
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Статистика модели - {user.username}")
        self.window.geometry("1000x700")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Данные для анализа
        self.training_samples = self.db.get_user_training_samples(user.id)
        self.model_info = self.model_manager.get_model_info(user.id)
        
        # Создание интерфейса
        self.create_widgets()
        
        # Загрузка данных
        self.load_real_statistics()
    
    def create_widgets(self):
        """Создание виджетов окна статистики"""
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка 1: Обзор
        self.create_overview_tab()
        
        # Вкладка 2: Анализ признаков
        self.create_features_tab()
        
        # Вкладка 3: Производительность (метрики безопасности)
        self.create_performance_tab()
        
        # Вкладка 4: ROC-кривая и метрики
        self.create_roc_tab()
        
        # Вкладка 5: Данные образцов
        self.create_samples_tab()
    
    def create_overview_tab(self):
        """Вкладка обзора модели"""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Обзор")
        
        # Информация о модели
        info_frame = ttk.LabelFrame(frame, text="Информация о модели", padding=15)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.overview_text = tk.Text(info_frame, height=8, width=70, font=(FONT_FAMILY, 10))
        self.overview_text.pack(fill=tk.BOTH, expand=True)
        
        # График распределения образцов по времени
        chart_frame = ttk.LabelFrame(frame, text="Распределение образцов по времени", padding=15)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig1, self.ax1 = plt.subplots(figsize=(10, 4))
        self.canvas1 = FigureCanvas(self.fig1, chart_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_features_tab(self):
        """Вкладка анализа признаков"""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Анализ признаков")
        
        # График признаков
        self.fig2, ((self.ax2a, self.ax2b), (self.ax2c, self.ax2d)) = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas2 = FigureCanvas(self.fig2, frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_performance_tab(self):
        """Вкладка производительности"""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Производительность")
        
        # Метрики безопасности
        metrics_frame = ttk.LabelFrame(frame, text="Метрики безопасности системы", padding=15)
        metrics_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.metrics_text = tk.Text(metrics_frame, height=12, width=80, font=(FONT_FAMILY, 10))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # График метрик
        chart_frame = ttk.LabelFrame(frame, text="Визуализация метрик", padding=15)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig3, self.ax3 = plt.subplots(figsize=(10, 5))
        self.canvas3 = FigureCanvas(self.fig3, chart_frame)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_roc_tab(self):
        """Вкладка ROC-анализа"""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="ROC-кривая")
        
        self.fig4, (self.ax4a, self.ax4b) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas4 = FigureCanvas(self.fig4, frame)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_samples_tab(self):
        """Вкладка данных образцов"""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="Данные образцов")
        
        # Таблица образцов
        table_frame = ttk.LabelFrame(frame, text="Обучающие образцы", padding=15)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создание Treeview
        columns = ('№', 'Время', 'Avg Dwell', 'Avg Flight', 'Скорость', 'Общее время')
        self.samples_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Настройка заголовков
        for col in columns:
            self.samples_tree.heading(col, text=col)
            self.samples_tree.column(col, width=120)
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.samples_tree.yview)
        self.samples_tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.samples_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_real_statistics(self):
        """Загрузка и расчет реальной статистики"""
        try:
            # 1. Обзор модели
            self.load_overview_stats()
            
            # 2. Анализ признаков
            self.load_features_analysis()
            
            # 3. Метрики производительности
            self.load_performance_metrics()
            
            # 4. ROC-анализ (только если есть sklearn)
            self.load_roc_analysis()
            
            # 5. Данные образцов
            self.load_samples_data()
            
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")
            import traceback
            traceback.print_exc()
    
    def load_overview_stats(self):
        """Загрузка обзорной статистики"""
        n_samples = len(self.training_samples)
        
        if n_samples == 0:
            self.overview_text.insert(tk.END, "Нет обучающих данных для анализа")
            return
        
        # Извлечение признаков для анализа
        features_data = []
        for sample in self.training_samples:
            if sample.features:
                features_data.append([
                    sample.features.get('avg_dwell_time', 0),
                    sample.features.get('avg_flight_time', 0),
                    sample.features.get('typing_speed', 0),
                    sample.features.get('total_typing_time', 0)
                ])
        
        if not features_data:
            self.overview_text.insert(tk.END, "Признаки не рассчитаны для образцов")
            return
        
        features_array = np.array(features_data)
        
        # Статистика
        overview_info = f"""ОБЗОР МОДЕЛИ ПОЛЬЗОВАТЕЛЯ: {self.user.username}

Основная информация:
• Количество обучающих образцов: {n_samples}
• Дата создания аккаунта: {self.user.created_at.strftime('%d.%m.%Y %H:%M')}
• Последний вход: {self.user.last_login.strftime('%d.%m.%Y %H:%M') if self.user.last_login else 'Никогда'}
• Статус модели: {'Обучена' if self.user.is_trained else 'Не обучена'}

Характеристики печати:
• Среднее время удержания клавиш: {np.mean(features_array[:, 0])*1000:.1f} ± {np.std(features_array[:, 0])*1000:.1f} мс
• Среднее время между клавишами: {np.mean(features_array[:, 1])*1000:.1f} ± {np.std(features_array[:, 1])*1000:.1f} мс  
• Средняя скорость печати: {np.mean(features_array[:, 2]):.1f} ± {np.std(features_array[:, 2]):.1f} клавиш/сек
• Среднее общее время: {np.mean(features_array[:, 3]):.1f} ± {np.std(features_array[:, 3]):.1f} сек

Вариативность (коэффициент вариации):
• Время удержания: {(np.std(features_array[:, 0])/np.mean(features_array[:, 0])*100):.1f}%
• Время между клавишами: {(np.std(features_array[:, 1])/np.mean(features_array[:, 1])*100):.1f}%
• Скорость печати: {(np.std(features_array[:, 2])/np.mean(features_array[:, 2])*100):.1f}%

Интерпретация:
• Низкая вариативность (<15%) = стабильная печать
• Средняя вариативность (15-30%) = обычная печать  
• Высокая вариативность (>30%) = нестабильная печать"""

        self.overview_text.insert(tk.END, overview_info)
        
        # График распределения по времени
        try:
            timestamps = [sample.timestamp for sample in self.training_samples]
            self.ax1.hist([t.hour for t in timestamps], bins=24, alpha=0.7, color='skyblue', edgecolor='black')
            self.ax1.set_xlabel('Час дня')
            self.ax1.set_ylabel('Количество образцов')
            self.ax1.set_title('Распределение сбора образцов по времени суток')
            self.ax1.grid(True, alpha=0.3)
            self.canvas1.draw()
        except Exception as e:
            print(f"Ошибка графика времени: {e}")
    
    def load_features_analysis(self):
        """Анализ признаков"""
        if not self.training_samples:
            return
        
        try:
            # Извлечение данных признаков
            features_data = []
            for sample in self.training_samples:
                if sample.features:
                    features_data.append([
                        sample.features.get('avg_dwell_time', 0) * 1000,  # в мс
                        sample.features.get('avg_flight_time', 0) * 1000,  # в мс
                        sample.features.get('typing_speed', 0),
                        sample.features.get('total_typing_time', 0)
                    ])
            
            if not features_data:
                return
            
            features_array = np.array(features_data)
            feature_names = ['Время удержания (мс)', 'Время между клавишами (мс)', 
                            'Скорость (клавиш/сек)', 'Общее время (сек)']
            
            # Четыре графика
            axes = [self.ax2a, self.ax2b, self.ax2c, self.ax2d]
            
            for i, (ax, name) in enumerate(zip(axes, feature_names)):
                data = features_array[:, i]
                
                # Гистограмма
                ax.hist(data, bins=min(10, len(data)//2 + 1), alpha=0.7, color=f'C{i}', edgecolor='black')
                ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Среднее: {np.mean(data):.2f}')
                ax.set_xlabel(name)
                ax.set_ylabel('Частота')
                ax.set_title(f'Распределение: {name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            self.fig2.tight_layout()
            self.canvas2.draw()
            
        except Exception as e:
            print(f"Ошибка анализа признаков: {e}")
    
    def load_performance_metrics(self):
        """Практический расчет метрик на основе реальных данных + симуляция"""
        try:
            # ДОБАВИТЬ ЭТУ ПРОВЕРКУ В НАЧАЛО МЕТОДА:
            if not hasattr(self, 'db') or self.db is None:
                # Получаем доступ к базе данных через parent window
                if hasattr(self.parent, 'db'):
                    self.db = self.parent.db
                elif hasattr(self.parent, 'parent') and hasattr(self.parent.parent, 'db'):
                    self.db = self.parent.parent.db
                else:
                    # Создаем подключение к базе
                    from utils.database import Database
                    self.db = Database()

            # Получаем реальные попытки аутентификации
            auth_attempts = self.db.get_auth_attempts(self.user.id, limit=50)
        
            if len(auth_attempts) < 3:
                metrics_info = f"""НЕДОСТАТОЧНО ДАННЫХ ДЛЯ АНАЛИЗА

    Найдено попыток: {len(auth_attempts)}
    Нужно минимум 3 попытки аутентификации.

    🚀 ЧТО СДЕЛАТЬ:
    1. Войдите в систему 3-5 раз с ПРАВИЛЬНОЙ скоростью
    2. Попробуйте войти 2-3 раза с МЕДЛЕННОЙ скоростью  
    3. Попробуйте войти 2-3 раза с БЫСТРОЙ скоростью
    4. Вернитесь к статистике

    💡 Это даст данные для расчета FAR/FRR/EER"""
            
                self.metrics_text.insert(tk.END, metrics_info)
                return

            print(f"\n📊 ПРАКТИЧЕСКИЙ АНАЛИЗ МЕТРИК")
            print(f"Реальных попыток: {len(auth_attempts)}")

            # Классификация ваших попыток по уверенности
            high_confidence = [a for a in auth_attempts if a['final_confidence'] >= 0.7]  # Точно вы
            medium_confidence = [a for a in auth_attempts if 0.4 <= a['final_confidence'] < 0.7]  # Сомнительно
            low_confidence = [a for a in auth_attempts if a['final_confidence'] < 0.4]  # Точно не вы

            # Анализ по категориям
            legitimate_attempts = high_confidence  # Это точно вы
            suspicious_attempts = medium_confidence + low_confidence  # Возможные имитации

            print(f"Высокая уверенность (>70%): {len(high_confidence)}")
            print(f"Средняя уверенность (40-70%): {len(medium_confidence)}")  
            print(f"Низкая уверенность (<40%): {len(low_confidence)}")

            # Расчет метрик для разных порогов
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            threshold_results = []

            for threshold in thresholds:
                # Считаем, что попытки с высокой уверенностью - это вы (legitimate)
                # Попытки с низкой уверенностью - это имитаторы (impostor)
            
                # Легитимные попытки (должны проходить)
                legit_passed = sum(1 for a in legitimate_attempts if a['final_confidence'] >= threshold)
                legit_total = len(legitimate_attempts) if legitimate_attempts else 1
            
                # "Имитаторы" (не должны проходить)  
                impostor_passed = sum(1 for a in suspicious_attempts if a['final_confidence'] >= threshold)
                impostor_total = len(suspicious_attempts) if suspicious_attempts else 1

                # Расчет метрик
                frr = ((legit_total - legit_passed) / legit_total) * 100  # Отклонили вас
                far = (impostor_passed / impostor_total) * 100  # Приняли имитатора
                eer = (far + frr) / 2
                accuracy = ((legit_passed + (impostor_total - impostor_passed)) / (legit_total + impostor_total)) * 100

                threshold_results.append({
                    'threshold': threshold,
                    'far': far,
                    'frr': frr, 
                    'eer': eer,
                    'accuracy': accuracy,
                    'legit_passed': legit_passed,
                    'legit_total': legit_total,
                    'impostor_passed': impostor_passed,
                    'impostor_total': impostor_total
                })

            # Текущий порог (обычно 0.75)
            current_threshold = 0.75
            current_result = min(threshold_results, key=lambda x: abs(x['threshold'] - current_threshold))
        
            # Оптимальный порог (минимальный EER)
            optimal_result = min(threshold_results, key=lambda x: x['eer'])

            # Статистика уверенности
            all_confidences = [a['final_confidence'] for a in auth_attempts]
            legit_confidences = [a['final_confidence'] for a in legitimate_attempts]
            suspicious_confidences = [a['final_confidence'] for a in suspicious_attempts] if suspicious_attempts else [0]

            # Анализ компонентов (KNN, Distance, Features)
            avg_knn = np.mean([a['knn_confidence'] for a in auth_attempts])
            avg_distance = np.mean([a['distance_score'] for a in auth_attempts])  
            avg_features = np.mean([a['feature_score'] for a in auth_attempts])

            metrics_info = f"""ПРАКТИЧЕСКИЕ МЕТРИКИ БЕЗОПАСНОСТИ:

📊 АНАЛИЗ ВАШИХ РЕАЛЬНЫХ ПОПЫТОК:

Классификация попыток по уверенности:
• Высокая уверенность (>70%): {len(high_confidence)} попыток - "Точно вы"
• Средняя уверенность (40-70%): {len(medium_confidence)} попыток - "Возможно вы"  
• Низкая уверенность (<40%): {len(low_confidence)} попыток - "Скорее не вы"

🎯 МЕТРИКИ ПРИ ТЕКУЩЕМ ПОРОГЕ ({current_threshold:.0%}):

FAR (False Acceptance Rate):
• Значение: {current_result['far']:.1f}%
• Интерпретация: Из {current_result['impostor_total']} подозрительных попыток система приняла {current_result['impostor_passed']}
• Статус: {'✅ ОТЛИЧНО' if current_result['far'] < 10 else '✅ ХОРОШО' if current_result['far'] < 25 else '⚠️ СРЕДНЕ' if current_result['far'] < 50 else '❌ ПЛОХО'}

FRR (False Rejection Rate):
• Значение: {current_result['frr']:.1f}%
• Интерпретация: Из {current_result['legit_total']} явно ваших попыток система отклонила {current_result['legit_total'] - current_result['legit_passed']}
• Статус: {'✅ ОТЛИЧНО' if current_result['frr'] < 15 else '✅ ХОРОШО' if current_result['frr'] < 30 else '⚠️ СРЕДНЕ' if current_result['frr'] < 50 else '❌ ПЛОХО'}

EER (Equal Error Rate):
• Значение: {current_result['eer']:.1f}%
• Статус: {'🏆 ОТЛИЧНО' if current_result['eer'] < 15 else '✅ ХОРОШО' if current_result['eer'] < 25 else '⚠️ СРЕДНЕ' if current_result['eer'] < 40 else '❌ ТРЕБУЕТ УЛУЧШЕНИЯ'}

Общая точность: {current_result['accuracy']:.1f}%

📈 АНАЛИЗ УВЕРЕННОСТИ:
• Средняя уверенность (все): {np.mean(all_confidences):.1%}
• Средняя уверенность (явно вы): {np.mean(legit_confidences):.1%}
• Средняя уверенность (подозрительные): {np.mean(suspicious_confidences):.1%}
• Разделимость: {abs(np.mean(legit_confidences) - np.mean(suspicious_confidences)):.1%}

🔧 КАК ФОРМИРУЕТСЯ ВАША УВЕРЕННОСТЬ ~{np.mean(legit_confidences):.0%}:
• KNN классификатор: {avg_knn:.1%} (основной алгоритм)
• Анализ расстояний: {avg_distance:.1%} (похожесть на обучающие данные)
• Анализ признаков: {avg_features:.1%} (разумность временных характеристик)

🎛️ ОПТИМИЗАЦИЯ:
• Рекомендуемый порог: {optimal_result['threshold']:.0%}
• EER при оптимальном пороге: {optimal_result['eer']:.1f}%
• {'📈 Повысьте порог для большей безопасности' if optimal_result['threshold'] > current_threshold else '📉 Понизьте порог для лучшей проходимости' if optimal_result['threshold'] < current_threshold else '✅ Текущий порог оптимален'}

📋 ИСТОРИЯ ПОСЛЕДНИХ ПОПЫТОК:"""

            # Последние попытки с классификацией
            recent_attempts = auth_attempts[:10]
            for i, attempt in enumerate(recent_attempts, 1):
                confidence = attempt['final_confidence']
                result_icon = "✅" if attempt['result'] else "❌"
            
                # Классификация типа попытки
                if confidence >= 0.7:
                    attempt_type = "🟢 Явно вы"
                elif confidence >= 0.4:
                    attempt_type = "🟡 Сомнительно"
                else:
                    attempt_type = "🔴 Скорее не вы"
            
                time_str = attempt['timestamp'].strftime('%d.%m %H:%M')
                metrics_info += f"\n{i:2d}. {time_str} | {result_icon} {confidence:.1%} | {attempt_type}"

            metrics_info += f"""

💡 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:
{self._interpret_practical_results(current_result, len(legitimate_attempts), len(suspicious_attempts))}

🚀 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ МЕТРИК:
{self._get_practical_recommendations(auth_attempts, current_result, optimal_result)}

⚠️ МЕТОДОЛОГИЯ: Анализ основан на ваших реальных попытках входа.
Попытки с высокой уверенностью считаются легитимными, с низкой - имитацией."""

            self.metrics_text.insert(tk.END, metrics_info)

            # График
            self._plot_practical_metrics(threshold_results, current_result, auth_attempts)

        except Exception as e:
            error_msg = f"Ошибка анализа: {str(e)}"
            self.metrics_text.insert(tk.END, error_msg)
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    def load_roc_analysis(self):
        """Реалистичный ROC-анализ"""
        try:
            # Проверяем наличие sklearn
            from sklearn.metrics import roc_curve, auc
        
            classifier = self.model_manager._get_user_model(self.user.id)
            if not classifier or not classifier.is_trained:
                self.ax4a.text(0.5, 0.5, 'Модель не обучена', 
                            ha='center', va='center', transform=self.ax4a.transAxes, fontsize=14)
                self.ax4b.text(0.5, 0.5, 'Модель не обучена', 
                            ha='center', va='center', transform=self.ax4b.transAxes, fontsize=14)
                self.canvas4.draw()
                return

            # Получаем обучающие данные
            X_positive = classifier.training_data
            n_samples = len(X_positive)
        
            print(f"\n📈 ROC-АНАЛИЗ с реалистичными данными")

            # Генерируем тестовые данные (как в метрике производительности)
            # 1. Реалистичные негативы
            X_negative_realistic = self._generate_realistic_negatives(X_positive, n_samples)
        
            # 2. Ваши вариации
            X_positive_variations = self._generate_user_variations(X_positive, int(n_samples * 0.3))
        
            # Объединяем все ваши данные (обучающие + вариации)
            X_all_positive = np.vstack([X_positive, X_positive_variations])
        
            # Создаем сбалансированный тестовый набор
            test_size = min(len(X_all_positive), len(X_negative_realistic))
        
            # Берем случайные подвыборки для баланса
            if len(X_all_positive) > test_size:
                pos_indices = np.random.choice(len(X_all_positive), test_size, replace=False)
                X_test_positive = X_all_positive[pos_indices]
            else:
                X_test_positive = X_all_positive
            
            if len(X_negative_realistic) > test_size:
                neg_indices = np.random.choice(len(X_negative_realistic), test_size, replace=False)
                X_test_negative = X_negative_realistic[neg_indices]
            else:
                X_test_negative = X_negative_realistic

            # Объединяем тестовые данные
            X_test = np.vstack([X_test_positive, X_test_negative])
            y_test = np.hstack([
                np.ones(len(X_test_positive)),   # 1 = ваши данные
                np.zeros(len(X_test_negative))   # 0 = чужие данные
            ])

            print(f"Тестовые данные для ROC: {len(X_test_positive)} ваших + {len(X_test_negative)} чужих")

            # Получаем вероятности классификации
            y_proba = classifier.model.predict_proba(X_test)
        
            # Убеждаемся, что у нас есть вероятности для класса 1
            if y_proba.shape[1] == 2:
                y_scores = y_proba[:, 1]  # вероятности для класса 1 (ваши данные)
            else:
                # Если модель предсказывает только один класс
                y_scores = classifier.model.decision_function(X_test)

            # Строим ROC кривую
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            # График 1: ROC кривая
            self.ax4a.clear()
            self.ax4a.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC кривая (AUC = {roc_auc:.3f})')
            self.ax4a.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                        label='Случайный классификатор (AUC = 0.5)')
        
            # Отмечаем оптимальную точку (максимизирует TPR - FPR)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            self.ax4a.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                        label=f'Оптимальный порог = {optimal_threshold:.3f}')
        
            self.ax4a.set_xlim([0.0, 1.0])
            self.ax4a.set_ylim([0.0, 1.05])
            self.ax4a.set_xlabel('False Positive Rate (FAR)')
            self.ax4a.set_ylabel('True Positive Rate (1 - FRR)')
            self.ax4a.set_title(f'ROC Кривая (AUC = {roc_auc:.3f})')
            self.ax4a.legend(loc="lower right", fontsize=9)
            self.ax4a.grid(True, alpha=0.3)

            # Добавляем текстовую интерпретацию AUC
            if roc_auc >= 0.9:
                auc_interpretation = "Отличная"
            elif roc_auc >= 0.8:
                auc_interpretation = "Хорошая"
            elif roc_auc >= 0.7:
                auc_interpretation = "Удовлетворительная"
            else:
                auc_interpretation = "Слабая"
        
            self.ax4a.text(0.6, 0.2, f'{auc_interpretation} модель', 
                        transform=self.ax4a.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

            # График 2: Распределение scores с более детальным анализом
            self.ax4b.clear()
        
            pos_scores = y_scores[y_test == 1]  # ваши оценки
            neg_scores = y_scores[y_test == 0]  # чужие оценки
        
            # Гистограммы с прозрачностью
            self.ax4b.hist(neg_scores, bins=25, alpha=0.6, label=f'Чужие (n={len(neg_scores)})', 
                        color='red', density=True, edgecolor='darkred')
            self.ax4b.hist(pos_scores, bins=25, alpha=0.6, label=f'Ваши (n={len(pos_scores)})', 
                        color='green', density=True, edgecolor='darkgreen')
        
            # Статистики распределений
            pos_mean, pos_std = np.mean(pos_scores), np.std(pos_scores)
            neg_mean, neg_std = np.mean(neg_scores), np.std(neg_scores)
        
            # Вертикальные линии для средних значений
            self.ax4b.axvline(pos_mean, color='darkgreen', linestyle='-', linewidth=2, 
                            alpha=0.8, label=f'Среднее ваших: {pos_mean:.3f}')
            self.ax4b.axvline(neg_mean, color='darkred', linestyle='-', linewidth=2, 
                            alpha=0.8, label=f'Среднее чужих: {neg_mean:.3f}')
        
            # Линия оптимального порога
            self.ax4b.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
                            alpha=0.8, label=f'Оптимальный порог: {optimal_threshold:.3f}')
        
            # Области ошибок
            if optimal_threshold < 1.0 and optimal_threshold > 0.0:
                # Область False Rejections (ваши данные ниже порога)
                x_fill_fr = np.linspace(min(y_scores), optimal_threshold, 100)
                y_fill_fr = np.histogram(pos_scores, bins=100, range=(min(y_scores), max(y_scores)), density=True)[0]
                x_bins = np.histogram(pos_scores, bins=100, range=(min(y_scores), max(y_scores)))[1]
                mask_fr = x_bins[:-1] <= optimal_threshold
                if np.any(mask_fr):
                    self.ax4b.fill_between(x_bins[:-1][mask_fr], 0, y_fill_fr[mask_fr], 
                                        alpha=0.3, color='orange', label='False Rejections')
            
                # Область False Acceptances (чужие данные выше порога)
                mask_fa = x_bins[:-1] >= optimal_threshold
                y_fill_fa = np.histogram(neg_scores, bins=100, range=(min(y_scores), max(y_scores)), density=True)[0]
                if np.any(mask_fa):
                    self.ax4b.fill_between(x_bins[:-1][mask_fa], 0, y_fill_fa[mask_fa], 
                                        alpha=0.3, color='yellow', label='False Acceptances')

            self.ax4b.set_xlabel('Уверенность классификатора')
            self.ax4b.set_ylabel('Плотность')
            self.ax4b.set_title('Распределение оценок классификатора')
            self.ax4b.legend(loc='upper right', fontsize=8)
            self.ax4b.grid(True, alpha=0.3)

            # Добавляем статистику разделимости
            denominator = np.sqrt((pos_std**2 + neg_std**2) / 2)
            if denominator > 0:
                separation = abs(pos_mean - neg_mean) / denominator
            else:
                separation = abs(pos_mean - neg_mean)
            self.ax4b.text(0.02, 0.98, 
                        f'Разделимость: {separation:.2f}\n'
                        f'Перекрытие: {min(np.max(neg_scores), np.max(pos_scores)) - max(np.min(neg_scores), np.min(pos_scores)):.3f}',
                        transform=self.ax4b.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

            self.canvas4.draw()
        
            print(f"ROC AUC: {roc_auc:.3f} ({auc_interpretation})")
            print(f"Оптимальный порог: {optimal_threshold:.3f}")
            print(f"Разделимость классов: {separation:.2f}")
        
        except ImportError:
            # sklearn не установлен
            self.ax4a.text(0.5, 0.5, 'scikit-learn не установлен\nROC-анализ недоступен', 
                        ha='center', va='center', transform=self.ax4a.transAxes, fontsize=14)
            self.ax4b.text(0.5, 0.5, 'Установите scikit-learn:\npip install scikit-learn', 
                        ha='center', va='center', transform=self.ax4b.transAxes, fontsize=14)
            self.canvas4.draw()
        except Exception as e:
            print(f"Ошибка ROC анализа: {e}")
            import traceback
            traceback.print_exc()
            self.ax4a.text(0.5, 0.5, f'Ошибка ROC анализа:\n{str(e)}', 
                        ha='center', va='center', transform=self.ax4a.transAxes, fontsize=12)
            self.canvas4.draw()
    
    def load_samples_data(self):
        """Загрузка данных образцов в таблицу"""
        try:
            # Очищаем таблицу
            for item in self.samples_tree.get_children():
                self.samples_tree.delete(item)
            
            # Заполняем данными
            for i, sample in enumerate(self.training_samples, 1):
                if sample.features:
                    self.samples_tree.insert('', 'end', values=(
                        i,
                        sample.timestamp.strftime('%d.%m %H:%M:%S'),
                        f"{sample.features.get('avg_dwell_time', 0)*1000:.1f} мс",
                        f"{sample.features.get('avg_flight_time', 0)*1000:.1f} мс", 
                        f"{sample.features.get('typing_speed', 0):.1f} кл/с",
                        f"{sample.features.get('total_typing_time', 0):.1f} с"
                    ))
        except Exception as e:
            print(f"Ошибка загрузки таблицы: {e}")



    def _generate_realistic_negatives(self, X_positive: np.ndarray, n_samples: int) -> np.ndarray:
        """Генерация РЕАЛИСТИЧНЫХ негативных примеров"""
        print(f"\n🎭 СОЗДАНИЕ РЕАЛИСТИЧНЫХ ИМИТАТОРОВ")
    
        # Анализ ваших данных
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
    
        print(f"Ваш профиль печати:")
        print(f"  Время удержания: {mean[0]*1000:.1f} ± {std[0]*1000:.1f} мс")
        print(f"  Время между клавишами: {mean[2]*1000:.1f} ± {std[2]*1000:.1f} мс")
        print(f"  Скорость: {mean[4]:.1f} ± {std[4]:.1f} клавиш/сек")
    
        realistic_samples = []
    
        # 1. Похожие пользователи (30%) - небольшие отличия
        similar_count = int(n_samples * 0.3)
        print(f"Создаем {similar_count} похожих пользователей...")
        for i in range(similar_count):
            # Отклонения в пределах 1.5-3 стандартных отклонений
            noise_factor = np.random.uniform(1.5, 3.0)
            noise = np.random.normal(0, std * noise_factor)
            sample = mean + noise
            # Ограничиваем снизу положительными значениями
            sample = np.maximum(sample, mean * 0.1)
            realistic_samples.append(sample)
    
        # 2. Умеренно отличающиеся (40%) 
        moderate_count = int(n_samples * 0.4)
        print(f"Создаем {moderate_count} умеренно отличающихся...")
        for i in range(moderate_count):
            # Систематические отличия в стиле печати
            factors = np.random.uniform(0.4, 2.5, size=6)  # каждый признак меняется индивидуально
            sample = mean * factors
            # Добавляем шум
            noise = np.random.normal(0, std * 0.8)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.05)
            realistic_samples.append(sample)
    
        # 3. Сильно отличающиеся (30%)
        different_count = n_samples - similar_count - moderate_count
        print(f"Создаем {different_count} сильно отличающихся...")
        for i in range(different_count):
            # Более драматичные, но реалистичные отличия
            if np.random.random() < 0.5:
                # Быстрые печатающие
                factors = np.array([
                    np.random.uniform(0.2, 0.7),    # короткое удержание
                    np.random.uniform(0.3, 0.8),    # низкая вариативность удержания
                    np.random.uniform(0.1, 0.5),    # короткие паузы
                    np.random.uniform(0.2, 0.7),    # низкая вариативность пауз
                    np.random.uniform(1.5, 4.0),    # высокая скорость
                    np.random.uniform(0.3, 0.8)     # меньше времени
                ])
            else:
                # Медленные печатающие
                factors = np.array([
                    np.random.uniform(1.5, 4.0),    # долгое удержание
                    np.random.uniform(1.2, 3.0),    # высокая вариативность
                    np.random.uniform(2.0, 6.0),    # долгие паузы
                    np.random.uniform(1.5, 4.0),    # высокая вариативность пауз
                    np.random.uniform(0.2, 0.7),    # низкая скорость
                    np.random.uniform(1.5, 4.0)     # больше времени
                ])
        
            sample = mean * factors
            noise = np.random.normal(0, std * 0.5)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.01)
            realistic_samples.append(sample)
    
        result = np.array(realistic_samples)
    
        # Проверяем реалистичность
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(result, X_positive)
        min_distances = np.min(distances, axis=1)
    
        print(f"\n📊 Статистика негативных образцов:")
        print(f"  Создано: {len(result)}")
        print(f"  Мин. расстояние до ваших: {np.min(min_distances):.3f}")
        print(f"  Среднее расстояние: {np.mean(min_distances):.3f}")
        print(f"  Макс. расстояние: {np.max(min_distances):.3f}")
    
        return result
    
    def _generate_user_variations(self, X_positive: np.ndarray, n_variations: int) -> np.ndarray:
        """Генерация ваших вариаций (когда печатаете по-разному)"""
        print(f"\n👤 СОЗДАНИЕ ВАШИХ ВАРИАЦИЙ")
    
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
    
        variations = []
    
        for i in range(n_variations):
            # Ваши естественные вариации - в пределах 2 стандартных отклонений
            variation_factor = np.random.uniform(0.8, 1.5)  # небольшие изменения
            noise = np.random.normal(0, std * variation_factor)
        
            # Иногда добавляем систематические изменения (усталость, спешка и т.д.)
            if np.random.random() < 0.3:
                # Эффект усталости - все замедляется
                systematic_factor = np.array([1.2, 1.1, 1.3, 1.2, 0.8, 1.25])
            elif np.random.random() < 0.3:
                # Эффект спешки - все ускоряется, но менее стабильно
                systematic_factor = np.array([0.8, 1.3, 0.7, 1.4, 1.3, 0.75])
            else:
                # Обычные небольшие вариации
                systematic_factor = np.random.uniform(0.9, 1.1, size=6)
        
            sample = mean * systematic_factor + noise
            sample = np.maximum(sample, mean * 0.1)  # не даем стать отрицательными
            variations.append(sample)
    
        print(f"  Создано вариаций ваших данных: {len(variations)}")
        return np.array(variations)
    

    def _plot_realistic_metrics(self, threshold_results, current_far, current_frr, current_eer):
        """График реалистичных метрик"""
        # Очищаем старый график
        self.ax3.clear()
    
        # График 1: Основные метрики
        metrics = ['FAR', 'FRR', 'EER']
        values = [current_far, current_frr, current_eer]
        colors = ['red', 'blue', 'green']
    
        bars = self.ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
        self.ax3.set_ylabel('Процент (%)')
        self.ax3.set_title('Реалистичные метрики безопасности')
        self.ax3.set_ylim(0, max(max(values) * 1.3, 10))
    
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
        # Добавляем линии рекомендуемых значений
        self.ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Хороший уровень (5%)')
        self.ax3.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Приемлемый уровень (15%)')
    
        self.ax3.legend(loc='upper right', fontsize=8)
        self.ax3.grid(True, alpha=0.3)
    
        self.canvas3.draw()


    def _get_metrics_interpretation(self, far, frr, eer, accuracy):
        """Интерпретация метрик"""
        interpretations = []
    
        if eer < 5:
            interpretations.append("🏆 ОТЛИЧНАЯ система - высокая безопасность и удобство")
        elif eer < 15:
            interpretations.append("✅ ХОРОШАЯ система - приемлемый баланс")
        elif eer < 25:
            interpretations.append("⚠️ СРЕДНЯЯ система - требует доработки")
        else:
            interpretations.append("❌ СЛАБАЯ система - нужно больше обучающих данных")
    
        if far > 10:
            interpretations.append("🚨 Высокий риск принятия имитаторов")
        elif far < 1:
            interpretations.append("🔒 Отличная защита от имитаторов")
    
        if frr > 20:
            interpretations.append("😤 Высокий риск отклонения легитимных пользователей")
        elif frr < 10:
            interpretations.append("👍 Хорошая проходимость для легитимных пользователей")
    
        return "\n".join(f"• {interp}" for interp in interpretations)
    

    def _generate_balanced_negatives(self, X_positive: np.ndarray, n_needed: int) -> np.ndarray:
        """Генерация СБАЛАНСИРОВАННЫХ негативных примеров"""
        print(f"\n🎭 СОЗДАНИЕ СБАЛАНСИРОВАННЫХ ИМИТАТОРОВ")
    
        mean = np.mean(X_positive, axis=0)
        std = np.std(X_positive, axis=0)
    
        # Обеспечиваем минимальную вариативность
        std = np.maximum(std, mean * 0.1)  # минимум 10% от среднего
    
        negatives = []
    
        # 40% - Близкие, но отличающиеся
        close_count = int(n_needed * 0.4)
        for i in range(close_count):
            # Изменяем 2-3 признака умеренно
            sample = mean.copy()
            features_to_change = np.random.choice(6, size=np.random.randint(2, 4), replace=False)
        
            for feat_idx in features_to_change:
                # Умеренные изменения: 0.6-1.8x от вашего среднего
                factor = np.random.choice([
                    np.random.uniform(0.6, 0.85),   # медленнее
                    np.random.uniform(1.15, 1.8)    # быстрее
                ])
                sample[feat_idx] = mean[feat_idx] * factor
        
            # Небольшой шум
            noise = np.random.normal(0, std * 0.4)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.05)
            negatives.append(sample)
    
        # 35% - Умеренно отличающиеся
        moderate_count = int(n_needed * 0.35)
        for i in range(moderate_count):
            # Систематические отличия в стиле
            if np.random.random() < 0.5:
                # Быстрый стиль
                factors = np.array([0.5, 0.7, 0.4, 0.6, 1.8, 0.6])
            else:
                # Медленный стиль
                factors = np.array([1.8, 1.5, 2.2, 1.8, 0.5, 2.0])
            
            sample = mean * factors
            noise = np.random.normal(0, std * 0.3)
            sample = sample + noise  
            sample = np.maximum(sample, mean * 0.02)
            negatives.append(sample)
    
        # 25% - Сильно отличающиеся, но реалистичные
        different_count = n_needed - close_count - moderate_count
        for i in range(different_count):
            # Экстремальные, но возможные стили
            if np.random.random() < 0.5:
                # Очень быстрые
                factors = np.array([0.2, 0.4, 0.15, 0.3, 3.5, 0.3])
            else:
                # Очень медленные
                factors = np.array([3.0, 2.5, 4.0, 3.5, 0.25, 4.0])
            
            sample = mean * factors
            noise = np.random.normal(0, std * 0.2)
            sample = sample + noise
            sample = np.maximum(sample, mean * 0.01)
            negatives.append(sample)
    
        result = np.array(negatives)
        print(f"  Создано {len(result)} сбалансированных негативных образцов")
    
        return result


    def _explain_confidence_calculation(self):
        """Объяснение расчета уверенности"""
        return """Система рассчитывает уверенность через KNN классификатор:

1. ОСНОВНОЙ МЕТОД (KNN):
   • Модель обучена на ваших + чужих данных
   • При тестировании возвращает вероятность 0-100%
   
2. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:
   • Расстояние до ваших обучающих образцов
   • Анализ отклонений признаков
   • Комбинирование оценок с весами
   
3. ФИНАЛЬНАЯ УВЕРЕННОСТЬ:
   • 80%+ = очень похоже на ваш стиль
   • 60-80% = похоже, но есть отличия  
   • 40-60% = сомнительно
   • <40% = скорее всего не вы"""
    

    def _interpret_results(self, far, frr, eer, separation):
        """Интерпретация результатов"""
        interpretations = []
    
        # Общая оценка
        if eer < 10:
            interpretations.append("🏆 ОТЛИЧНАЯ биометрическая система")
        elif eer < 20:
            interpretations.append("✅ ХОРОШАЯ система для практического использования")
        elif eer < 35:
            interpretations.append("⚠️ ПРИЕМЛЕМАЯ система, но можно улучшить")
        else:
            interpretations.append("❌ СЛАБАЯ система, нужно больше данных")
    
        # Безопасность
        if far < 5:
            interpretations.append("🔒 Высокая защита от имитаторов")
        elif far < 15:
            interpretations.append("🛡️ Приемлемая защита от имитаторов")
        else:
            interpretations.append("⚠️ Риск принятия имитаторов - рассмотрите повышение порога")
    
        # Удобство
        if frr < 10:
            interpretations.append("👍 Отличная проходимость для вас")
        elif frr < 25:
            interpretations.append("✅ Хорошая проходимость, редкие отказы")
        else:
            interpretations.append("😤 Частые отказы - рассмотрите понижение порога")
    
        # Разделимость
        if separation > 0.3:
            interpretations.append("📊 Отличная разделимость классов")
        elif separation > 0.15:
            interpretations.append("📈 Хорошая разделимость классов")
        else:
            interpretations.append("📉 Слабая разделимость - нужно больше обучающих данных")
    
        return "\n".join(f"• {interp}" for interp in interpretations)
    

    def _plot_stable_metrics(self, results_by_threshold, standard_result):
        """График стабильных метрик"""
        self.ax3.clear()
    
        # График основных метрик при стандартном пороге
        metrics = ['FAR', 'FRR', 'EER']
        values = [standard_result['far'], standard_result['frr'], standard_result['eer']]
        colors = ['red', 'blue', 'green']
    
        bars = self.ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
        self.ax3.set_ylabel('Процент (%)')
        self.ax3.set_title('Метрики безопасности (порог 50%)')
        self.ax3.set_ylim(0, max(max(values) * 1.2, 20))
    
        # Значения на столбцах
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
        # Референсные линии
        self.ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Хороший уровень')
        self.ax3.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Приемлемый уровень')
    
        self.ax3.legend(loc='upper right', fontsize=10)
        self.ax3.grid(True, alpha=0.3)
    
        self.canvas3.draw()


    def _generate_normalized_negatives(self, X_positive: np.ndarray, n_needed: int) -> np.ndarray:
        """Генерация негативов в нормализованном пространстве"""
        # В нормализованном пространстве среднее ~0, std ~1
    
        negatives = []
    
        # 30% - близкие конкуренты (сложные для различения)
        close_count = int(n_needed * 0.3)
        for i in range(close_count):
            # Небольшие отклонения от нормализованного нуля
            sample = np.random.normal(0, 0.8, size=6)  # немного ближе к центру
            negatives.append(sample)
    
        # 40% - умеренно отличающиеся
        moderate_count = int(n_needed * 0.4)  
        for i in range(moderate_count):
            # Средние отклонения
            sample = np.random.normal(0, 1.5, size=6)
            negatives.append(sample)
    
        # 30% - сильно отличающиеся
        far_count = n_needed - close_count - moderate_count
        for i in range(far_count):
            # Большие отклонения  
            sample = np.random.normal(0, 2.5, size=6)
            negatives.append(sample)
    
        return np.array(negatives)
    


    def _interpret_system_quality(self, far, frr, eer, separation):
        """Интерпретация качества системы"""
        quality_notes = []
    
        if eer < 15 and separation > 0.2:
            quality_notes.append("🏆 Отличная биометрическая система!")
            quality_notes.append("✅ Готова для практического использования")
        elif eer < 25:
            quality_notes.append("✅ Хорошая система с приемлемыми характеристиками")
        else:
            quality_notes.append("⚠️ Система требует улучшения")
        
        if far < 10:
            quality_notes.append("🔒 Хорошая защита от имитаторов")
        else:
            quality_notes.append("⚠️ Рассмотрите повышение порога безопасности")
        
        if frr < 20:
            quality_notes.append("👍 Удобство использования на высоком уровне")
        else:
            quality_notes.append("😅 Возможны частые отказы - рассмотрите понижение порога")
    
        return "\n".join(f"• {note}" for note in quality_notes)
    


    def _plot_clean_metrics(self, results_by_threshold, standard_result):
        """Чистый график метрик"""
        self.ax3.clear()
    
        metrics = ['FAR', 'FRR', 'EER']
        values = [standard_result['far'], standard_result['frr'], standard_result['eer']]
        colors = ['red', 'blue', 'green']
    
        bars = self.ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        self.ax3.set_ylabel('Процент (%)')
        self.ax3.set_title('Метрики безопасности системы')
        self.ax3.set_ylim(0, max(max(values) * 1.2, 25))
    
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
        self.ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Хороший уровень')
        self.ax3.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Приемлемый уровень')
    
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        self.canvas3.draw()



    def _interpret_real_performance(self, all_attempts, successful, failed):
        """Интерпретация реальной производительности"""
        success_rate = len(successful) / len(all_attempts) * 100
    
        if success_rate >= 90:
            return "🏆 Отличная система! Очень высокая точность распознавания."
        elif success_rate >= 80:
            return "✅ Хорошая система с высокой надежностью."
        elif success_rate >= 70:
            return "👍 Приемлемая производительность, возможны улучшения."
        elif success_rate >= 60:
            return "⚠️ Средняя производительность, рекомендуется настройка."
        else:
            return "❌ Низкая производительность, требуется диагностика."
        

    def _get_recommendations(self, attempts, threshold_analysis, recent_success_rate):
        """Рекомендации по улучшению"""
        recommendations = []
    
        current_threshold = attempts[0]['threshold_used'] if attempts else 0.5
        optimal = min(threshold_analysis, key=lambda x: x['eer'])
    
        if abs(optimal['threshold'] - current_threshold) > 0.1:
            if optimal['threshold'] < current_threshold:
                recommendations.append("🔽 Рассмотрите понижение порога для лучшей проходимости")
            else:
                recommendations.append("🔼 Рассмотрите повышение порога для большей безопасности")
    
        if recent_success_rate < 70:
            recommendations.append("📚 Попробуйте переобучить модель с новыми образцами")
            recommendations.append("⌨️ Убедитесь, что печатаете в том же стиле, что и при обучении")
    
        if len(attempts) < 20:
            recommendations.append("🔄 Больше попыток аутентификации улучшат точность статистики")
    
        confidences = [a['final_confidence'] for a in attempts]
        if np.std(confidences) > 0.25:
            recommendations.append("📊 Высокая вариативность уверенности - возможно, стиль печати нестабилен")
    
        return "\n".join(f"• {rec}" for rec in recommendations) if recommendations else "• Система работает оптимально!"
    

    def _plot_real_performance(self, attempts, threshold_analysis):
        """График реальной производительности"""
        self.ax3.clear()
    
        # График 1: Распределение уверенности
        confidences = [a['final_confidence'] for a in attempts]
        results = [a['result'] for a in attempts]
    
        success_conf = [c for c, r in zip(confidences, results) if r]
        fail_conf = [c for c, r in zip(confidences, results) if not r]
    
        if success_conf:
            self.ax3.hist(success_conf, bins=10, alpha=0.7, color='green', 
                        label=f'Успешные ({len(success_conf)})', density=True)
        if fail_conf:
            self.ax3.hist(fail_conf, bins=10, alpha=0.7, color='red',
                        label=f'Отклоненные ({len(fail_conf)})', density=True)
    
        # Текущий порог
        current_threshold = attempts[0]['threshold_used'] if attempts else 0.5
        self.ax3.axvline(current_threshold, color='black', linestyle='--', 
                        label=f'Текущий порог ({current_threshold:.1%})')
    
        self.ax3.set_xlabel('Уверенность системы')
        self.ax3.set_ylabel('Плотность')
        self.ax3.set_title('Распределение реальных попыток аутентификации')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
    
        self.canvas3.draw()



    def _interpret_practical_results(self, result, legit_count, suspicious_count):
        """Интерпретация практических результатов"""
        interpretations = []
    
        if result['eer'] < 15:
            interpretations.append("🏆 Отличная биометрическая система!")
        elif result['eer'] < 25:
            interpretations.append("✅ Хорошая система с приемлемыми характеристиками")
        elif result['eer'] < 40:
            interpretations.append("⚠️ Система работает, но есть место для улучшений")
        else:
            interpretations.append("❌ Система требует серьезной доработки")

        if result['far'] < 15:
            interpretations.append("🔒 Хорошая защита от имитаторов")
        else:
            interpretations.append("⚠️ Высокий риск принятия имитаторов")

        if result['frr'] < 20:
            interpretations.append("👍 Хорошая проходимость для легитимного пользователя")
        else:
            interpretations.append("😤 Частые отказы легитимному пользователю")

        if legit_count < 3:
            interpretations.append("📊 Мало данных для надежной оценки - нужно больше успешных входов")
    
        if suspicious_count < 2:
            interpretations.append("🎭 Мало данных об имитации - попробуйте войти с разной скоростью")

        return "\n".join(f"• {interp}" for interp in interpretations)
    

    def _get_practical_recommendations(self, attempts, current_result, optimal_result):
        """Практические рекомендации"""
        recommendations = []
    
        # Рекомендации по сбору данных
        legit_attempts = [a for a in attempts if a['final_confidence'] >= 0.7]
        suspicious_attempts = [a for a in attempts if a['final_confidence'] < 0.4]
    
        if len(legit_attempts) < 5:
            recommendations.append("📈 Сделайте еще 3-5 успешных входов с правильной скоростью печати")
    
        if len(suspicious_attempts) < 3:
            recommendations.append("🎭 Попробуйте войти медленно/быстро для имитации 'чужого' стиля")
    
        # Рекомендации по настройке
        if abs(optimal_result['threshold'] - 0.75) > 0.1:
            if optimal_result['threshold'] < 0.75:
                recommendations.append(f"🔽 Понизьте порог до {optimal_result['threshold']:.0%} для лучшей проходимости")
            else:
                recommendations.append(f"🔼 Повысьте порог до {optimal_result['threshold']:.0%} для большей безопасности")
    
        # Рекомендации по качеству
        if current_result['eer'] > 25:
            recommendations.append("🔄 Рассмотрите переобучение модели с новыми образцами")
            recommendations.append("⌨️ Убедитесь в стабильности стиля печати")
    
        if len(attempts) < 10:
            recommendations.append("📊 Больше попыток аутентификации улучшат точность метрик")
    
        return "\n".join(f"• {rec}" for rec in recommendations) if recommendations else "• Система работает оптимально!"
    

    def _plot_practical_metrics(self, threshold_results, current_result, attempts):
        """График практических метрик"""
        self.ax3.clear()
    
        # График основных метрик
        metrics = ['FAR', 'FRR', 'EER']
        values = [current_result['far'], current_result['frr'], current_result['eer']]
        colors = ['red', 'blue', 'green']
    
        bars = self.ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
        self.ax3.set_ylabel('Процент (%)')
        self.ax3.set_title(f'Практические метрики (порог {current_result["threshold"]:.0%})')
        self.ax3.set_ylim(0, max(max(values) * 1.2, 25))
    
        # Референсные линии
        self.ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Хороший уровень')
        self.ax3.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Приемлемый уровень')
    
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        self.canvas3.draw()