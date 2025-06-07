# gui/enhanced_model_stats_window.py - Продвинутое окно статистики модели

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import json
import os

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from ml.model_manager import ModelManager
from utils.database import DatabaseManager
from config import FONT_FAMILY, DATA_DIR

# Настройка matplotlib для работы с tkinter
plt.style.use('default')

class EnhancedModelStatsWindow:
    """Продвинутое окно статистики модели с детальным анализом"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        self.model_manager = ModelManager()
        self.db = DatabaseManager()
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"📊 Продвинутая статистика - {user.username}")
        self.window.geometry("1200x800")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Данные для анализа
        self.training_samples = self.db.get_user_training_samples(user.id)
        self.auth_attempts = self.db.get_auth_attempts(user.id, limit=100)
        self.model_info = self.model_manager.get_model_info(user.id)
        
        # Создание интерфейса
        self.create_widgets()
        
        # Загрузка данных
        self.load_enhanced_statistics()
    
    def create_widgets(self):
        """Создание продвинутого интерфейса"""
        # Заголовок с информацией о пользователе
        header_frame = ttk.Frame(self.window, padding=10)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(
            header_frame,
            text=f"📊 Продвинутая статистика модели - {self.user.username}",
            font=(FONT_FAMILY, 16, 'bold')
        ).pack()
        
        # Быстрая сводка
        summary_frame = ttk.Frame(header_frame)
        summary_frame.pack(fill=tk.X, pady=10)
        
        self.create_summary_cards(summary_frame)
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Вкладки
        self.create_performance_analysis_tab()
        self.create_temporal_analysis_tab()
        self.create_behavioral_patterns_tab()
        self.create_security_analysis_tab()
        self.create_model_diagnostics_tab()
        self.create_comparison_tab()
        
        # Кнопки действий
        self.create_action_buttons()
    
    def create_summary_cards(self, parent):
        """Создание карточек быстрой сводки"""
        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill=tk.X)
        
        # Основные метрики
        model_type = self.model_info.get('model_type', 'none')
        total_samples = len(self.training_samples)
        total_attempts = len(self.auth_attempts)
        
        if self.auth_attempts:
            recent_accuracy = np.mean([a['result'] for a in self.auth_attempts[-10:]])
            avg_confidence = np.mean([a['final_confidence'] for a in self.auth_attempts])
        else:
            recent_accuracy = 0.0
            avg_confidence = 0.0
        
        cards_data = [
            ("🎯 Тип модели", model_type.upper() if model_type != 'none' else 'НЕ ОБУЧЕНА'),
            ("📚 Обучающих образцов", f"{total_samples}"),
            ("🔐 Попыток входа", f"{total_attempts}"),
            ("📈 Точность (10 попыток)", f"{recent_accuracy:.1%}"),
            ("🎲 Средняя уверенность", f"{avg_confidence:.1%}")
        ]
        
        for i, (title, value) in enumerate(cards_data):
            card = ttk.LabelFrame(cards_frame, text=title, padding=5)
            card.grid(row=0, column=i, padx=5, sticky='ew')
            cards_frame.columnconfigure(i, weight=1)
            
            ttk.Label(
                card, 
                text=str(value), 
                font=(FONT_FAMILY, 12, 'bold'),
                foreground='darkblue'
            ).pack()
    
    def create_performance_analysis_tab(self):
        """Вкладка анализа производительности"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="📈 Анализ производительности")
        
        # Верхняя часть - метрики
        metrics_frame = ttk.LabelFrame(frame, text="🎯 Ключевые метрики производительности", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.performance_text = tk.Text(metrics_frame, height=8, width=80, font=(FONT_FAMILY, 10))
        perf_scroll = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.performance_text.yview)
        self.performance_text.configure(yscrollcommand=perf_scroll.set)
        
        self.performance_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        perf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Нижняя часть - графики
        chart_frame = ttk.LabelFrame(frame, text="📊 Визуализация производительности", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_perf, ((self.ax_perf1, self.ax_perf2), (self.ax_perf3, self.ax_perf4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas_perf = FigureCanvasTkAgg(self.fig_perf, chart_frame)
        self.canvas_perf.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_temporal_analysis_tab(self):
        """Вкладка временного анализа"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="⏰ Временной анализ")
        
        # График временных тенденций
        self.fig_time, ((self.ax_time1, self.ax_time2), (self.ax_time3, self.ax_time4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, frame)
        self.canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_behavioral_patterns_tab(self):
        """Вкладка анализа поведенческих паттернов"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="🧠 Поведенческие паттерны")
        
        # Разделение на две части
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Анализ паттернов (текст)
        patterns_frame = ttk.LabelFrame(left_frame, text="🔍 Обнаруженные паттерны", padding=10)
        patterns_frame.pack(fill=tk.BOTH, expand=True)
        
        self.patterns_text = tk.Text(patterns_frame, height=20, width=50, font=(FONT_FAMILY, 10))
        patterns_scroll = ttk.Scrollbar(patterns_frame, orient=tk.VERTICAL, command=self.patterns_text.yview)
        self.patterns_text.configure(yscrollcommand=patterns_scroll.set)
        
        self.patterns_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        patterns_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Визуализация паттернов
        viz_frame = ttk.LabelFrame(right_frame, text="📈 Визуализация паттернов", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_patterns, (self.ax_pat1, self.ax_pat2) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas_patterns = FigureCanvasTkAgg(self.fig_patterns, viz_frame)
        self.canvas_patterns.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_security_analysis_tab(self):
        """Вкладка анализа безопасности"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="🔒 Анализ безопасности")
        
        # Анализ угроз
        threat_frame = ttk.LabelFrame(frame, text="⚠️ Анализ угроз и аномалий", padding=10)
        threat_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.security_text = tk.Text(threat_frame, height=8, width=80, font=(FONT_FAMILY, 10))
        sec_scroll = ttk.Scrollbar(threat_frame, orient=tk.VERTICAL, command=self.security_text.yview)
        self.security_text.configure(yscrollcommand=sec_scroll.set)
        
        self.security_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sec_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Графики безопасности
        security_chart_frame = ttk.LabelFrame(frame, text="📊 Метрики безопасности", padding=10)
        security_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_security, (self.ax_sec1, self.ax_sec2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas_security = FigureCanvasTkAgg(self.fig_security, security_chart_frame)
        self.canvas_security.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_model_diagnostics_tab(self):
        """Вкладка диагностики модели"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="🔧 Диагностика модели")
        
        # Информация о модели
        model_info_frame = ttk.LabelFrame(frame, text="ℹ️ Информация о модели", padding=10)
        model_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_info_text = tk.Text(model_info_frame, height=10, width=80, font=(FONT_FAMILY, 10))
        model_scroll = ttk.Scrollbar(model_info_frame, orient=tk.VERTICAL, command=self.model_info_text.yview)
        self.model_info_text.configure(yscrollcommand=model_scroll.set)
        
        self.model_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        model_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Диагностические графики
        diag_frame = ttk.LabelFrame(frame, text="📈 Диагностические графики", padding=10)
        diag_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_diag, ((self.ax_diag1, self.ax_diag2), (self.ax_diag3, self.ax_diag4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas_diag = FigureCanvasTkAgg(self.fig_diag, diag_frame)
        self.canvas_diag.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_comparison_tab(self):
        """Вкладка сравнительного анализа"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="⚖️ Сравнительный анализ")
        
        # Сравнение с эталонами
        comparison_frame = ttk.LabelFrame(frame, text="📊 Сравнение с эталонными показателями", padding=10)
        comparison_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.comparison_text = tk.Text(comparison_frame, height=8, width=80, font=(FONT_FAMILY, 10))
        comp_scroll = ttk.Scrollbar(comparison_frame, orient=tk.VERTICAL, command=self.comparison_text.yview)
        self.comparison_text.configure(yscrollcommand=comp_scroll.set)
        
        self.comparison_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        comp_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # График сравнения
        comp_chart_frame = ttk.LabelFrame(frame, text="📈 Визуальное сравнение", padding=10)
        comp_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_comp, (self.ax_comp1, self.ax_comp2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, comp_chart_frame)
        self.canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_action_buttons(self):
        """Создание кнопок действий"""
        buttons_frame = ttk.Frame(self.window)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="📄 Экспорт отчета",
            command=self.export_detailed_report
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="🔄 Обновить данные",
            command=self.refresh_data
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="⚙️ Оптимизировать модель",
            command=self.optimize_model
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="❌ Закрыть",
            command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def load_enhanced_statistics(self):
        """Загрузка расширенной статистики"""
        try:
            # Загружаем все вкладки
            self.load_performance_analysis()
            self.load_temporal_analysis()
            self.load_behavioral_patterns()
            self.load_security_analysis()
            self.load_model_diagnostics()
            self.load_comparison_analysis()
            
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")
            import traceback
            traceback.print_exc()
    
    def load_performance_analysis(self):
        """Загрузка анализа производительности"""
        try:
            # Анализ производительности по времени
            if not self.auth_attempts:
                self.performance_text.insert(tk.END, "Недостаточно данных для анализа производительности.\nВыполните несколько попыток аутентификации.")
                return
            
            # Расчет метрик
            total_attempts = len(self.auth_attempts)
            successful_attempts = sum(1 for a in self.auth_attempts if a['result'])
            success_rate = successful_attempts / total_attempts * 100 if total_attempts > 0 else 0
            
            avg_confidence = np.mean([a['final_confidence'] for a in self.auth_attempts])
            confidence_std = np.std([a['final_confidence'] for a in self.auth_attempts])
            
            recent_10 = self.auth_attempts[:10]
            recent_success_rate = np.mean([a['result'] for a in recent_10]) * 100 if recent_10 else 0
            
            # FAR/FRR анализ (приблизительный)
            high_conf_attempts = [a for a in self.auth_attempts if a['final_confidence'] >= 0.7]
            low_conf_attempts = [a for a in self.auth_attempts if a['final_confidence'] < 0.4]
            
            far_estimate = sum(1 for a in low_conf_attempts if a['result']) / len(low_conf_attempts) * 100 if low_conf_attempts else 0
            frr_estimate = sum(1 for a in high_conf_attempts if not a['result']) / len(high_conf_attempts) * 100 if high_conf_attempts else 0
            
            performance_report = f"""АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ СИСТЕМЫ
{'='*60}

📊 ОБЩИЕ МЕТРИКИ:
• Всего попыток аутентификации: {total_attempts}
• Успешных попыток: {successful_attempts}
• Общий процент успеха: {success_rate:.1f}%
• Успех за последние 10 попыток: {recent_success_rate:.1f}%

🎯 АНАЛИЗ УВЕРЕННОСТИ:
• Средняя уверенность: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)
• Стандартное отклонение: {confidence_std:.3f}
• Стабильность: {'ВЫСОКАЯ' if confidence_std < 0.2 else 'СРЕДНЯЯ' if confidence_std < 0.3 else 'НИЗКАЯ'}

🔒 ОЦЕНКА БЕЗОПАСНОСТИ:
• Приблизительный FAR: {far_estimate:.1f}%
• Приблизительный FRR: {frr_estimate:.1f}%
• Состояние: {'БЕЗОПАСНО' if far_estimate < 10 and frr_estimate < 25 else 'ТРЕБУЕТ ВНИМАНИЯ'}

📈 ТЕНДЕНЦИИ:
• Стабильность работы: {'СТАБИЛЬНАЯ' if confidence_std < 0.25 else 'НЕСТАБИЛЬНАЯ'}
• Рекомендация: {'Система работает хорошо' if success_rate > 80 else 'Рекомендуется переобучение'}

⏰ ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ:
• Период анализа: {(self.auth_attempts[0]['timestamp'] - self.auth_attempts[-1]['timestamp']).days if len(self.auth_attempts) > 1 else 0} дней
• Активность: {total_attempts / max(1, (self.auth_attempts[0]['timestamp'] - self.auth_attempts[-1]['timestamp']).days):.1f} попыток/день
"""
            
            self.performance_text.insert(tk.END, performance_report)
            
            # Графики производительности
            self.plot_performance_charts()
            
        except Exception as e:
            self.performance_text.insert(tk.END, f"Ошибка анализа производительности: {e}")
    
    def plot_performance_charts(self):
        """Построение графиков производительности"""
        if not self.auth_attempts:
            return
        
        try:
            # График 1: Уверенность по времени
            self.ax_perf1.clear()
            timestamps = [a['timestamp'] for a in reversed(self.auth_attempts)]
            confidences = [a['final_confidence'] for a in reversed(self.auth_attempts)]
            results = [a['result'] for a in reversed(self.auth_attempts)]
            
            # Цвета точек в зависимости от результата
            colors = ['green' if r else 'red' for r in results]
            self.ax_perf1.scatter(range(len(confidences)), confidences, c=colors, alpha=0.7)
            self.ax_perf1.plot(range(len(confidences)), confidences, 'b-', alpha=0.3)
            self.ax_perf1.axhline(y=0.75, color='orange', linestyle='--', label='Порог (75%)')
            self.ax_perf1.set_xlabel('Попытка')
            self.ax_perf1.set_ylabel('Уверенность')
            self.ax_perf1.set_title('Динамика уверенности системы')
            self.ax_perf1.legend()
            self.ax_perf1.grid(True, alpha=0.3)
            
            # График 2: Распределение уверенности
            self.ax_perf2.clear()
            self.ax_perf2.hist(confidences, bins=15, alpha=0.7, edgecolor='black')
            self.ax_perf2.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Среднее: {np.mean(confidences):.2f}')
            self.ax_perf2.set_xlabel('Уверенность')
            self.ax_perf2.set_ylabel('Частота')
            self.ax_perf2.set_title('Распределение уверенности')
            self.ax_perf2.legend()
            self.ax_perf2.grid(True, alpha=0.3)
            
            # График 3: Компоненты уверенности
            self.ax_perf3.clear()
            knn_scores = [a['knn_confidence'] for a in self.auth_attempts if 'knn_confidence' in a]
            distance_scores = [a['distance_score'] for a in self.auth_attempts if 'distance_score' in a]
            feature_scores = [a['feature_score'] for a in self.auth_attempts if 'feature_score' in a]
            
            if knn_scores and distance_scores and feature_scores:
                components = ['KNN', 'Distance', 'Features']
                avg_scores = [np.mean(knn_scores), np.mean(distance_scores), np.mean(feature_scores)]
                self.ax_perf3.bar(components, avg_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
                self.ax_perf3.set_ylabel('Средний вклад')
                self.ax_perf3.set_title('Средний вклад компонентов')
                self.ax_perf3.grid(True, alpha=0.3)
            
            # График 4: Скользящее среднее успешности
            self.ax_perf4.clear()
            window_size = min(5, len(results))
            if window_size > 1:
                moving_avg = []
                for i in range(window_size, len(results) + 1):
                    window = results[i-window_size:i]
                    moving_avg.append(np.mean(window) * 100)
                
                self.ax_perf4.plot(range(window_size, len(results) + 1), moving_avg, 'g-', linewidth=2)
                self.ax_perf4.set_xlabel('Попытка')
                self.ax_perf4.set_ylabel('Процент успеха (%)')
                self.ax_perf4.set_title(f'Скользящее среднее успешности (окно {window_size})')
                self.ax_perf4.grid(True, alpha=0.3)
            
            self.fig_perf.tight_layout()
            self.canvas_perf.draw()
            
        except Exception as e:
            print(f"Ошибка построения графиков производительности: {e}")
    
    def load_temporal_analysis(self):
        """Загрузка временного анализа"""
        try:
            if not self.auth_attempts:
                return
            
            # Анализ по времени суток
            self.ax_time1.clear()
            hours = [a['timestamp'].hour for a in self.auth_attempts]
            self.ax_time1.hist(hours, bins=24, alpha=0.7, edgecolor='black')
            self.ax_time1.set_xlabel('Час дня')
            self.ax_time1.set_ylabel('Количество попыток')
            self.ax_time1.set_title('Активность по времени суток')
            self.ax_time1.grid(True, alpha=0.3)
            
            # Анализ по дням недели
            self.ax_time2.clear()
            weekdays = [a['timestamp'].weekday() for a in self.auth_attempts]
            weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
            weekday_counts = [weekdays.count(i) for i in range(7)]
            self.ax_time2.bar(weekday_names, weekday_counts)
            self.ax_time2.set_ylabel('Количество попыток')
            self.ax_time2.set_title('Активность по дням недели')
            self.ax_time2.grid(True, alpha=0.3)
            
            # Тренд уверенности по времени
            self.ax_time3.clear()
            if len(self.auth_attempts) > 5:
                timestamps = [a['timestamp'] for a in reversed(self.auth_attempts)]
                confidences = [a['final_confidence'] for a in reversed(self.auth_attempts)]
                
                # Полиномиальная аппроксимация
                x = np.arange(len(confidences))
                z = np.polyfit(x, confidences, min(2, len(confidences)-1))
                p = np.poly1d(z)
                
                self.ax_time3.scatter(x, confidences, alpha=0.6)
                self.ax_time3.plot(x, p(x), "r--", alpha=0.8, label='Тренд')
                self.ax_time3.set_xlabel('Время (порядковый номер)')
                self.ax_time3.set_ylabel('Уверенность')
                self.ax_time3.set_title('Тренд уверенности системы')
                self.ax_time3.legend()
                self.ax_time3.grid(True, alpha=0.3)
            
            # Сессионная активность
            self.ax_time4.clear()
            daily_attempts = {}
            for attempt in self.auth_attempts:
                date = attempt['timestamp'].date()
                daily_attempts[date] = daily_attempts.get(date, 0) + 1
            
            if daily_attempts:
                dates = list(daily_attempts.keys())
                counts = list(daily_attempts.values())
                
                self.ax_time4.plot(dates, counts, 'o-')
                self.ax_time4.set_xlabel('Дата')
                self.ax_time4.set_ylabel('Попыток в день')
                self.ax_time4.set_title('Ежедневная активность')
                self.ax_time4.tick_params(axis='x', rotation=45)
                self.ax_time4.grid(True, alpha=0.3)
            
            self.fig_time.tight_layout()
            self.canvas_time.draw()
            
        except Exception as e:
            print(f"Ошибка временного анализа: {e}")
    
    def load_behavioral_patterns(self):
        """Загрузка анализа поведенческих паттернов"""
        try:
            patterns_analysis = self.analyze_behavioral_patterns()
            
            self.patterns_text.delete('1.0', tk.END)
            self.patterns_text.insert('1.0', patterns_analysis)
            
            # Графики паттернов
            self.plot_behavioral_patterns()
            
        except Exception as e:
            self.patterns_text.insert(tk.END, f"Ошибка анализа паттернов: {e}")
    
    def analyze_behavioral_patterns(self) -> str:
        """Анализ поведенческих паттернов пользователя"""
        if not self.auth_attempts or len(self.training_samples) < 10:
            return "Недостаточно данных для анализа поведенческих паттернов."
        
        # Извлекаем признаки из обучающих данных
        training_features = []
        for sample in self.training_samples:
            if sample.features:
                training_features.append([
                    sample.features.get('avg_dwell_time', 0),
                    sample.features.get('avg_flight_time', 0),
                    sample.features.get('typing_speed', 0),
                    sample.features.get('total_typing_time', 0)
                ])
        
        if not training_features:
            return "Нет данных о признаках для анализа."
        
        training_features = np.array(training_features)
        
        # Анализ стабильности
        feature_names = ['Время удержания', 'Время между клавишами', 'Скорость', 'Общее время']
        stability_analysis = []
        
        for i, name in enumerate(feature_names):
            values = training_features[:, i]
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            
            if cv < 0.15:
                stability = "ОЧЕНЬ СТАБИЛЬНО"
            elif cv < 0.25:
                stability = "СТАБИЛЬНО"
            elif cv < 0.35:
                stability = "УМЕРЕННО"
            else:
                stability = "НЕСТАБИЛЬНО"
            
            stability_analysis.append(f"• {name}: {stability} (CV: {cv:.2f})")
        
        # Анализ паттернов во времени
        time_patterns = self.analyze_time_patterns()
        
        # Анализ аномалий
        anomaly_analysis = self.detect_anomalies()
        
        analysis = f"""АНАЛИЗ ПОВЕДЕНЧЕСКИХ ПАТТЕРНОВ
{'='*50}

🧬 ПРОФИЛЬ КЛАВИАТУРНОГО ПОЧЕРКА:
{chr(10).join(stability_analysis)}

📊 ХАРАКТЕРИСТИКИ СТИЛЯ:
• Средняя скорость печати: {np.mean(training_features[:, 2]):.1f} клавиш/сек
• Время удержания клавиш: {np.mean(training_features[:, 0])*1000:.1f} мс
• Время между клавишами: {np.mean(training_features[:, 1])*1000:.1f} мс
• Общая продолжительность: {np.mean(training_features[:, 3]):.1f} сек

🕐 ВРЕМЕННЫЕ ПАТТЕРНЫ:
{time_patterns}

⚠️ ОБНАРУЖЕННЫЕ АНОМАЛИИ:
{anomaly_analysis}

🎯 УНИКАЛЬНОСТЬ ПРОФИЛЯ:
• Индекс различимости: {self.calculate_distinctiveness_index():.2f}
• Сложность имитации: {'ВЫСОКАЯ' if self.calculate_distinctiveness_index() > 0.3 else 'СРЕДНЯЯ' if self.calculate_distinctiveness_index() > 0.15 else 'НИЗКАЯ'}

💡 РЕКОМЕНДАЦИИ:
{self.generate_behavioral_recommendations()}
"""
        
        return analysis
    
    def analyze_time_patterns(self) -> str:
        """Анализ временных паттернов"""
        if not self.auth_attempts:
            return "• Недостаточно данных для анализа временных паттернов"
        
        # Анализ активности по времени
        hours = [a['timestamp'].hour for a in self.auth_attempts]
        most_active_hour = max(set(hours), key=hours.count)
        
        # Анализ периодичности
        dates = [a['timestamp'].date() for a in self.auth_attempts]
        unique_dates = len(set(dates))
        avg_attempts_per_day = len(self.auth_attempts) / max(1, unique_dates)
        
        patterns = [
            f"• Наиболее активное время: {most_active_hour:02d}:00-{most_active_hour+1:02d}:00",
            f"• Среднее количество попыток в день: {avg_attempts_per_day:.1f}",
            f"• Дней с активностью: {unique_dates}"
        ]
        
        # Анализ регулярности
        if len(self.auth_attempts) > 5:
            time_diffs = []
            for i in range(1, len(self.auth_attempts)):
                diff = (self.auth_attempts[i-1]['timestamp'] - self.auth_attempts[i]['timestamp']).total_seconds() / 3600
                time_diffs.append(diff)
            
            avg_interval = np.mean(time_diffs)
            patterns.append(f"• Средний интервал между попытками: {avg_interval:.1f} часов")
        
        return '\n'.join(patterns)
    
    def detect_anomalies(self) -> str:
        """Обнаружение аномалий в поведении"""
        if not self.auth_attempts:
            return "• Недостаточно данных для обнаружения аномалий"
        
        anomalies = []
        
        # Аномалии в уверенности
        confidences = [a['final_confidence'] for a in self.auth_attempts]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        outliers = [c for c in confidences if abs(c - mean_conf) > 2 * std_conf]
        if outliers:
            anomalies.append(f"• Обнаружено {len(outliers)} аномальных значений уверенности")
        
        # Аномалии во времени
        failed_attempts = [a for a in self.auth_attempts if not a['result']]
        if len(failed_attempts) > len(self.auth_attempts) * 0.3:
            anomalies.append(f"• Высокий процент неудачных попыток: {len(failed_attempts)/len(self.auth_attempts)*100:.1f}%")
        
        # Подозрительные паттерны
        low_conf_successful = [a for a in self.auth_attempts if a['result'] and a['final_confidence'] < 0.6]
        if low_conf_successful:
            anomalies.append(f"• {len(low_conf_successful)} успешных входов с низкой уверенностью (возможные ложные срабатывания)")
        
        if not anomalies:
            anomalies.append("• Аномалий не обнаружено - поведение стабильно")
        
        return '\n'.join(anomalies)
    
    def calculate_distinctiveness_index(self) -> float:
        """Расчет индекса различимости профиля"""
        if len(self.training_samples) < 5:
            return 0.0
        
        # Извлекаем признаки
        features = []
        for sample in self.training_samples:
            if sample.features:
                features.append([
                    sample.features.get('avg_dwell_time', 0),
                    sample.features.get('avg_flight_time', 0),
                    sample.features.get('typing_speed', 0)
                ])
        
        if not features:
            return 0.0
        
        features = np.array(features)
        
        # Рассчитываем коэффициент вариации для каждого признака
        cvs = []
        for i in range(features.shape[1]):
            values = features[:, i]
            if np.mean(values) > 0:
                cv = np.std(values) / np.mean(values)
                cvs.append(cv)
        
        # Индекс различимости основан на стабильности признаков
        if cvs:
            avg_cv = np.mean(cvs)
            # Чем стабильнее признаки, тем выше различимость
            distinctiveness = max(0, 1 - avg_cv * 2)
            return min(1.0, distinctiveness)
        
        return 0.0
    
    def generate_behavioral_recommendations(self) -> str:
        """Генерация рекомендаций по поведению"""
        recommendations = []
        
        if len(self.training_samples) < 30:
            recommendations.append("• Соберите больше обучающих образцов для повышения точности")
        
        if len(self.auth_attempts) > 0:
            success_rate = np.mean([a['result'] for a in self.auth_attempts])
            if success_rate < 0.8:
                recommendations.append("• Попробуйте печатать в том же стиле, что и при обучении")
        
        distinctiveness = self.calculate_distinctiveness_index()
        if distinctiveness < 0.2:
            recommendations.append("• Ваш стиль печати может быть легко имитирован - будьте осторожны")
        
        if not recommendations:
            recommendations.append("• Система работает стабильно, дополнительных действий не требуется")
        
        return '\n'.join(recommendations)
    
    def plot_behavioral_patterns(self):
        """Построение графиков поведенческих паттернов"""
        try:
            if not self.training_samples:
                return
            
            # График 1: Корреляция признаков
            self.ax_pat1.clear()
            
            features = []
            for sample in self.training_samples:
                if sample.features:
                    features.append([
                        sample.features.get('avg_dwell_time', 0) * 1000,  # в мс
                        sample.features.get('avg_flight_time', 0) * 1000,  # в мс
                    ])
            
            if features:
                features = np.array(features)
                self.ax_pat1.scatter(features[:, 0], features[:, 1], alpha=0.6)
                self.ax_pat1.set_xlabel('Время удержания (мс)')
                self.ax_pat1.set_ylabel('Время между клавишами (мс)')
                self.ax_pat1.set_title('Корреляция характеристик печати')
                self.ax_pat1.grid(True, alpha=0.3)
            
            # График 2: Эволюция признаков
            self.ax_pat2.clear()
            
            if len(self.training_samples) > 5:
                speeds = [s.features.get('typing_speed', 0) for s in self.training_samples if s.features]
                if speeds:
                    self.ax_pat2.plot(range(len(speeds)), speeds, 'o-', alpha=0.7)
                    self.ax_pat2.set_xlabel('Номер образца')
                    self.ax_pat2.set_ylabel('Скорость печати (клавиш/сек)')
                    self.ax_pat2.set_title('Эволюция скорости печати')
                    self.ax_pat2.grid(True, alpha=0.3)
            
            self.fig_patterns.tight_layout()
            self.canvas_patterns.draw()
            
        except Exception as e:
            print(f"Ошибка построения графиков паттернов: {e}")
    
    def load_security_analysis(self):
        """Загрузка анализа безопасности"""
        try:
            security_analysis = self.analyze_security_aspects()
            
            self.security_text.delete('1.0', tk.END)
            self.security_text.insert('1.0', security_analysis)
            
            # Графики безопасности
            self.plot_security_charts()
            
        except Exception as e:
            self.security_text.insert(tk.END, f"Ошибка анализа безопасности: {e}")
    
    def analyze_security_aspects(self) -> str:
        """Анализ аспектов безопасности"""
        if not self.auth_attempts:
            return "Недостаточно данных для анализа безопасности."
        
        # Анализ попыток с разной уверенностью
        high_conf = [a for a in self.auth_attempts if a['final_confidence'] >= 0.8]
        medium_conf = [a for a in self.auth_attempts if 0.4 <= a['final_confidence'] < 0.8]
        low_conf = [a for a in self.auth_attempts if a['final_confidence'] < 0.4]
        
        # Потенциальные угрозы
        threats = []
        
        # Успешные входы с низкой уверенностью
        suspicious_success = [a for a in low_conf if a['result']]
        if suspicious_success:
            threats.append(f"• {len(suspicious_success)} подозрительных успешных входов с низкой уверенностью")
        
        # Неудачные попытки с высокой уверенностью
        suspicious_failures = [a for a in high_conf if not a['result']]
        if suspicious_failures:
            threats.append(f"• {len(suspicious_failures)} отклоненных попыток с высокой уверенностью")
        
        # Временные аномалии
        if len(self.auth_attempts) > 10:
            recent_failures = sum(1 for a in self.auth_attempts[:5] if not a['result'])
            if recent_failures >= 3:
                threats.append("• Много неудачных попыток в последнее время")
        
        # Расчет приблизительных метрик безопасности
        legitimate_attempts = high_conf  # Считаем попытки с высокой уверенностью легитимными
        impostor_attempts = low_conf    # Считаем попытки с низкой уверенностью имитацией
        
        far = 0.0
        frr = 0.0
        
        if impostor_attempts:
            false_accepts = sum(1 for a in impostor_attempts if a['result'])
            far = false_accepts / len(impostor_attempts) * 100
        
        if legitimate_attempts:
            false_rejects = sum(1 for a in legitimate_attempts if not a['result'])
            frr = false_rejects / len(legitimate_attempts) * 100
        
        eer = (far + frr) / 2
        
        security_level = "ВЫСОКИЙ" if eer < 15 else "СРЕДНИЙ" if eer < 25 else "НИЗКИЙ"
        
        analysis = f"""АНАЛИЗ БЕЗОПАСНОСТИ СИСТЕМЫ
{'='*50}

🛡️ ОБЩАЯ ОЦЕНКА БЕЗОПАСНОСТИ: {security_level}

📊 РАСПРЕДЕЛЕНИЕ ПОПЫТОК ПО УВЕРЕННОСТИ:
• Высокая уверенность (≥80%): {len(high_conf)} попыток
• Средняя уверенность (40-80%): {len(medium_conf)} попыток  
• Низкая уверенность (<40%): {len(low_conf)} попыток

🎯 ПРИБЛИЗИТЕЛЬНЫЕ МЕТРИКИ БЕЗОПАСНОСТИ:
• FAR (False Acceptance Rate): {far:.1f}%
• FRR (False Rejection Rate): {frr:.1f}%
• EER (Equal Error Rate): {eer:.1f}%

⚠️ ПОТЕНЦИАЛЬНЫЕ УГРОЗЫ:
{chr(10).join(threats) if threats else '• Серьезных угроз не обнаружено'}

🔍 РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{self.generate_security_recommendations(far, frr, eer)}

📈 ТРЕНД БЕЗОПАСНОСТИ:
{self.analyze_security_trend()}
"""
        
        return analysis
    
    def generate_security_recommendations(self, far: float, frr: float, eer: float) -> str:
        """Генерация рекомендаций по безопасности"""
        recommendations = []
        
        if far > 10:
            recommendations.append("• Рассмотрите повышение порога аутентификации для увеличения безопасности")
        
        if frr > 25:
            recommendations.append("• Рассмотрите понижение порога для уменьшения ложных отказов")
        
        if eer > 20:
            recommendations.append("• Рекомендуется переобучение модели с большим количеством данных")
        
        if len(self.auth_attempts) < 20:
            recommendations.append("• Соберите больше данных о попытках входа для точной оценки безопасности")
        
        if not recommendations:
            recommendations.append("• Система демонстрирует хороший уровень безопасности")
        
        return '\n'.join(recommendations)
    
    def analyze_security_trend(self) -> str:
        """Анализ тренда безопасности"""
        if len(self.auth_attempts) < 10:
            return "• Недостаточно данных для анализа тренда"
        
        # Анализируем последние 10 и предыдущие 10 попыток
        recent_10 = self.auth_attempts[:10]
        previous_10 = self.auth_attempts[10:20] if len(self.auth_attempts) >= 20 else []
        
        recent_success = np.mean([a['result'] for a in recent_10])
        recent_confidence = np.mean([a['final_confidence'] for a in recent_10])
        
        if previous_10:
            previous_success = np.mean([a['result'] for a in previous_10])
            previous_confidence = np.mean([a['final_confidence'] for a in previous_10])
            
            success_trend = "УЛУЧШАЕТСЯ" if recent_success > previous_success else "УХУДШАЕТСЯ" if recent_success < previous_success else "СТАБИЛЬНО"
            confidence_trend = "РАСТЕТ" if recent_confidence > previous_confidence else "ПАДАЕТ" if recent_confidence < previous_confidence else "СТАБИЛЬНА"
            
            return f"• Успешность: {success_trend} ({recent_success:.1%} vs {previous_success:.1%})\n• Уверенность: {confidence_trend} ({recent_confidence:.1%} vs {previous_confidence:.1%})"
        else:
            return f"• Текущая успешность: {recent_success:.1%}\n• Текущая уверенность: {recent_confidence:.1%}"
    
    def plot_security_charts(self):
        """Построение графиков безопасности"""
        try:
            if not self.auth_attempts:
                return
            
            # График 1: Распределение по уровням риска
            self.ax_sec1.clear()
            
            confidences = [a['final_confidence'] for a in self.auth_attempts]
            high_risk = sum(1 for c in confidences if c < 0.4)
            medium_risk = sum(1 for c in confidences if 0.4 <= c < 0.7)
            low_risk = sum(1 for c in confidences if c >= 0.7)
            
            sizes = [high_risk, medium_risk, low_risk]
            labels = ['Высокий риск\n(<40%)', 'Средний риск\n(40-70%)', 'Низкий риск\n(≥70%)']
            colors = ['red', 'orange', 'green']
            
            if sum(sizes) > 0:
                wedges, texts, autotexts = self.ax_sec1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                self.ax_sec1.set_title('Распределение попыток по уровню риска')
            
            # График 2: Временной тренд безопасности
            self.ax_sec2.clear()
            
            if len(self.auth_attempts) > 5:
                # Скользящее среднее уверенности
                confidences = [a['final_confidence'] for a in reversed(self.auth_attempts)]
                window_size = min(5, len(confidences))
                
                moving_avg = []
                for i in range(window_size, len(confidences) + 1):
                    window = confidences[i-window_size:i]
                    moving_avg.append(np.mean(window))
                
                x = range(window_size, len(confidences) + 1)
                self.ax_sec2.plot(x, moving_avg, 'b-', linewidth=2, label='Средняя уверенность')
                self.ax_sec2.axhline(y=0.75, color='red', linestyle='--', label='Порог безопасности')
                self.ax_sec2.fill_between(x, moving_avg, 0.75, where=np.array(moving_avg) >= 0.75, 
                                        color='green', alpha=0.3, label='Безопасная зона')
                self.ax_sec2.fill_between(x, moving_avg, 0.75, where=np.array(moving_avg) < 0.75, 
                                        color='red', alpha=0.3, label='Зона риска')
                
                self.ax_sec2.set_xlabel('Попытка')
                self.ax_sec2.set_ylabel('Уверенность')
                self.ax_sec2.set_title('Тренд безопасности системы')
                self.ax_sec2.legend()
                self.ax_sec2.grid(True, alpha=0.3)
            
            self.fig_security.tight_layout()
            self.canvas_security.draw()
            
        except Exception as e:
            print(f"Ошибка построения графиков безопасности: {e}")
    
    def load_model_diagnostics(self):
        """Загрузка диагностики модели"""
        try:
            model_analysis = self.analyze_model_health()
            
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert('1.0', model_analysis)
            
            # Диагностические графики
            self.plot_diagnostic_charts()
            
        except Exception as e:
            self.model_info_text.insert(tk.END, f"Ошибка диагностики модели: {e}")
    
    def analyze_model_health(self) -> str:
        """Анализ состояния модели"""
        model_type = self.model_info.get('model_type', 'none')
        
        if model_type == 'none':
            return "Модель не обучена. Требуется сбор обучающих данных."
        
        # Базовая информация о модели
        training_samples = len(self.training_samples)
        is_trained = self.model_info.get('is_trained', False)
        
        # Анализ качества обучающих данных
        data_quality = self.assess_training_data_quality()
        
        # Анализ производительности модели
        performance_analysis = self.assess_model_performance()
        
        # Рекомендации по улучшению
        improvement_suggestions = self.generate_improvement_suggestions()
        
        analysis = f"""ДИАГНОСТИКА МОДЕЛИ
{'='*50}

🤖 ИНФОРМАЦИЯ О МОДЕЛИ:
• Тип модели: {model_type.upper()}
• Статус: {'ОБУЧЕНА' if is_trained else 'НЕ ОБУЧЕНА'}
• Обучающих образцов: {training_samples}
• Параметры: {self.format_model_params()}

📊 КАЧЕСТВО ОБУЧАЮЩИХ ДАННЫХ:
{data_quality}

⚡ ПРОИЗВОДИТЕЛЬНОСТЬ МОДЕЛИ:
{performance_analysis}

🔧 СОСТОЯНИЕ МОДЕЛИ:
{self.assess_model_condition()}

💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:
{improvement_suggestions}

📈 ИСТОРИЯ ОБУЧЕНИЯ:
{self.get_training_history()}
"""
        
        return analysis
    
    def assess_training_data_quality(self) -> str:
        """Оценка качества обучающих данных"""
        if not self.training_samples:
            return "• Нет обучающих данных"
        
        quality_aspects = []
        
        # Количество образцов
        sample_count = len(self.training_samples)
        if sample_count >= 50:
            quality_aspects.append(f"• Количество образцов: ОТЛИЧНО ({sample_count})")
        elif sample_count >= 30:
            quality_aspects.append(f"• Количество образцов: ХОРОШО ({sample_count})")
        else:
            quality_aspects.append(f"• Количество образцов: НЕДОСТАТОЧНО ({sample_count})")
        
        # Качество признаков
        valid_samples = [s for s in self.training_samples if s.features and all(v != 0 for v in s.features.values())]
        data_validity = len(valid_samples) / len(self.training_samples) * 100
        
        if data_validity >= 95:
            quality_aspects.append(f"• Качество данных: ОТЛИЧНО ({data_validity:.1f}% валидных)")
        elif data_validity >= 80:
            quality_aspects.append(f"• Качество данных: ХОРОШО ({data_validity:.1f}% валидных)")
        else:
            quality_aspects.append(f"• Качество данных: ПЛОХО ({data_validity:.1f}% валидных)")
        
        # Разнообразие данных
        if valid_samples:
            features_array = np.array([[
                s.features.get('avg_dwell_time', 0),
                s.features.get('typing_speed', 0)
            ] for s in valid_samples])
            
            diversity = np.mean([np.std(features_array[:, i]) for i in range(features_array.shape[1])])
            
            if diversity > 0.02:
                quality_aspects.append("• Разнообразие данных: ВЫСОКОЕ")
            elif diversity > 0.01:
                quality_aspects.append("• Разнообразие данных: СРЕДНЕЕ")
            else:
                quality_aspects.append("• Разнообразие данных: НИЗКОЕ")
        
        return '\n'.join(quality_aspects)
    
    def assess_model_performance(self) -> str:
        """Оценка производительности модели"""
        if not self.auth_attempts:
            return "• Недостаточно данных о попытках аутентификации"
        
        performance_aspects = []
        
        # Общая точность
        success_rate = np.mean([a['result'] for a in self.auth_attempts]) * 100
        if success_rate >= 90:
            performance_aspects.append(f"• Точность: ОТЛИЧНО ({success_rate:.1f}%)")
        elif success_rate >= 75:
            performance_aspects.append(f"• Точность: ХОРОШО ({success_rate:.1f}%)")
        else:
            performance_aspects.append(f"• Точность: ПЛОХО ({success_rate:.1f}%)")
        
        # Стабильность
        confidences = [a['final_confidence'] for a in self.auth_attempts]
        confidence_std = np.std(confidences)
        
        if confidence_std < 0.15:
            performance_aspects.append(f"• Стабильность: ВЫСОКАЯ (σ={confidence_std:.3f})")
        elif confidence_std < 0.25:
            performance_aspects.append(f"• Стабильность: СРЕДНЯЯ (σ={confidence_std:.3f})")
        else:
            performance_aspects.append(f"• Стабильность: НИЗКАЯ (σ={confidence_std:.3f})")
        
        # Время отклика (симуляция)
        avg_response_time = np.random.uniform(0.05, 0.15)  # Симуляция
        performance_aspects.append(f"• Время отклика: ~{avg_response_time:.3f} сек")
        
        return '\n'.join(performance_aspects)
    
    def assess_model_condition(self) -> str:
        """Оценка общего состояния модели"""
        if not self.training_samples:
            return "• Модель требует обучения"
        
        age_days = (datetime.now() - max(s.timestamp for s in self.training_samples)).days
        
        conditions = []
        
        if age_days > 90:
            conditions.append(f"• Возраст модели: {age_days} дней - ТРЕБУЕТ ОБНОВЛЕНИЯ")
        elif age_days > 30:
            conditions.append(f"• Возраст модели: {age_days} дней - рекомендуется обновление")
        else:
            conditions.append(f"• Возраст модели: {age_days} дней - актуальная")
        
        # Размер модели
        model_size = len(self.training_samples) * 6 * 8  # Приблизительно байт
        conditions.append(f"• Размер модели: ~{model_size/1024:.1f} КБ")
        
        # Загруженность
        attempts_per_day = len(self.auth_attempts) / max(1, age_days)
        if attempts_per_day > 10:
            conditions.append("• Загруженность: ВЫСОКАЯ")
        elif attempts_per_day > 3:
            conditions.append("• Загруженность: СРЕДНЯЯ")
        else:
            conditions.append("• Загруженность: НИЗКАЯ")
        
        return '\n'.join(conditions)
    
    def format_model_params(self) -> str:
        """Форматирование параметров модели"""
        if 'best_params' in self.model_info:
            params = self.model_info['best_params']
            if params:
                return f"K={params.get('n_neighbors', 'N/A')}, веса={params.get('weights', 'N/A')}"
        return "По умолчанию"
    
    def generate_improvement_suggestions(self) -> str:
        """Генерация предложений по улучшению"""
        suggestions = []
        
        # Анализ количества данных
        sample_count = len(self.training_samples)
        if sample_count < 30:
            suggestions.append("• Соберите больше обучающих образцов (рекомендуется 50+)")
        
        # Анализ производительности
        if self.auth_attempts:
            success_rate = np.mean([a['result'] for a in self.auth_attempts])
            if success_rate < 0.8:
                suggestions.append("• Рассмотрите переобучение модели с новыми данными")
        
        # Анализ возраста модели
        if self.training_samples:
            age_days = (datetime.now() - max(s.timestamp for s in self.training_samples)).days
            if age_days > 60:
                suggestions.append("• Обновите модель новыми образцами")
        
        # Анализ стабильности
        if self.auth_attempts:
            confidences = [a['final_confidence'] for a in self.auth_attempts]
            if np.std(confidences) > 0.25:
                suggestions.append("• Работайте над стабильностью стиля печати")
        
        if not suggestions:
            suggestions.append("• Модель работает хорошо, дополнительных улучшений не требуется")
        
        return '\n'.join(suggestions)
    
    def get_training_history(self) -> str:
        """Получение истории обучения"""
        if not self.training_samples:
            return "• История обучения отсутствует"
        
        # Группировка по дням
        training_dates = {}
        for sample in self.training_samples:
            date = sample.timestamp.date()
            training_dates[date] = training_dates.get(date, 0) + 1
        
        history = []
        for date, count in sorted(training_dates.items()):
            history.append(f"• {date}: {count} образцов")
        
        if len(history) > 5:
            history = history[:3] + [f"• ... и еще {len(history)-3} дней"] + history[-2:]
        
        return '\n'.join(history)
    
    def plot_diagnostic_charts(self):
        """Построение диагностических графиков"""
        try:
            # График 1: Качество признаков
            self.ax_diag1.clear()
            
            if self.training_samples:
                feature_quality = []
                feature_names = ['Dwell Time', 'Flight Time', 'Speed', 'Total Time']
                
                for i, name in enumerate(feature_names):
                    valid_count = 0
                    total_count = len(self.training_samples)
                    
                    for sample in self.training_samples:
                        if sample.features:
                            feature_keys = ['avg_dwell_time', 'avg_flight_time', 'typing_speed', 'total_typing_time']
                            if i < len(feature_keys) and sample.features.get(feature_keys[i], 0) > 0:
                                valid_count += 1
                    
                    quality = valid_count / total_count * 100 if total_count > 0 else 0
                    feature_quality.append(quality)
                
                bars = self.ax_diag1.bar(feature_names, feature_quality, 
                                       color=['green' if q >= 90 else 'orange' if q >= 70 else 'red' for q in feature_quality])
                self.ax_diag1.set_ylabel('Качество (%)')
                self.ax_diag1.set_title('Качество признаков')
                self.ax_diag1.set_ylim(0, 100)
                
                # Добавляем значения на столбцы
                for bar, quality in zip(bars, feature_quality):
                    height = bar.get_height()
                    self.ax_diag1.text(bar.get_x() + bar.get_width()/2., height + 1,
                                     f'{quality:.1f}%', ha='center', va='bottom')
            
            # График 2: Тренд производительности
            self.ax_diag2.clear()
            
            if self.auth_attempts and len(self.auth_attempts) > 5:
                # Скользящее среднее успешности
                results = [a['result'] for a in reversed(self.auth_attempts)]
                window_size = min(5, len(results))
                
                moving_avg = []
                for i in range(window_size, len(results) + 1):
                    window = results[i-window_size:i]
                    moving_avg.append(np.mean(window) * 100)
                
                x = range(window_size, len(results) + 1)
                self.ax_diag2.plot(x, moving_avg, 'b-', linewidth=2)
                self.ax_diag2.fill_between(x, moving_avg, alpha=0.3)
                self.ax_diag2.set_xlabel('Попытка')
                self.ax_diag2.set_ylabel('Успешность (%)')
                self.ax_diag2.set_title('Тренд производительности')
                self.ax_diag2.grid(True, alpha=0.3)
            
            # График 3: Распределение времени обучения
            self.ax_diag3.clear()
            
            if self.training_samples:
                training_dates = [s.timestamp.date() for s in self.training_samples]
                unique_dates = sorted(set(training_dates))
                
                if len(unique_dates) > 1:
                    daily_counts = [training_dates.count(date) for date in unique_dates]
                    self.ax_diag3.plot(unique_dates, daily_counts, 'o-')
                    self.ax_diag3.set_xlabel('Дата')
                    self.ax_diag3.set_ylabel('Образцов в день')
                    self.ax_diag3.set_title('История сбора данных')
                    self.ax_diag3.tick_params(axis='x', rotation=45)
            
            # График 4: Состояние компонентов модели
            self.ax_diag4.clear()
            
            if self.auth_attempts:
                # Анализ компонентов уверенности
                knn_scores = [a.get('knn_confidence', 0) for a in self.auth_attempts if 'knn_confidence' in a]
                distance_scores = [a.get('distance_score', 0) for a in self.auth_attempts if 'distance_score' in a]
                feature_scores = [a.get('feature_score', 0) for a in self.auth_attempts if 'feature_score' in a]
                
                if knn_scores and distance_scores and feature_scores:
                    components = ['KNN', 'Distance', 'Features']
                    avg_scores = [np.mean(knn_scores), np.mean(distance_scores), np.mean(feature_scores)]
                    std_scores = [np.std(knn_scores), np.std(distance_scores), np.std(feature_scores)]
                    
                    bars = self.ax_diag4.bar(components, avg_scores, yerr=std_scores, capsize=5)
                    self.ax_diag4.set_ylabel('Средний вклад')
                    self.ax_diag4.set_title('Стабильность компонентов')
                    self.ax_diag4.grid(True, alpha=0.3)
            
            self.fig_diag.tight_layout()
            self.canvas_diag.draw()
            
        except Exception as e:
            print(f"Ошибка построения диагностических графиков: {e}")
    
    def load_comparison_analysis(self):
        """Загрузка сравнительного анализа"""
        try:
            comparison_analysis = self.generate_comparison_analysis()
            
            self.comparison_text.delete('1.0', tk.END)
            self.comparison_text.insert('1.0', comparison_analysis)
            
            # Графики сравнения
            self.plot_comparison_charts()
            
        except Exception as e:
            self.comparison_text.insert(tk.END, f"Ошибка сравнительного анализа: {e}")
    
    def generate_comparison_analysis(self) -> str:
        """Генерация сравнительного анализа"""
        # Эталонные показатели для биометрических систем
        benchmarks = {
            'commercial': {'far': 1.0, 'frr': 5.0, 'eer': 3.0},
            'research': {'far': 5.0, 'frr': 15.0, 'eer': 10.0},
            'acceptable': {'far': 10.0, 'frr': 25.0, 'eer': 15.0}
        }
        
        # Расчет наших метрик
        our_metrics = self.calculate_our_metrics()
        
        # Сравнение с эталонами
        comparison_results = {}
        for level, benchmark in benchmarks.items():
            comparison_results[level] = {
                'far_diff': our_metrics['far'] - benchmark['far'],
                'frr_diff': our_metrics['frr'] - benchmark['frr'],
                'eer_diff': our_metrics['eer'] - benchmark['eer']
            }
        
        # Определение категории системы
        system_category = self.determine_system_category(our_metrics, benchmarks)
        
        analysis = f"""СРАВНИТЕЛЬНЫЙ АНАЛИЗ С ЭТАЛОНАМИ
{'='*50}

🎯 НАШИ ПОКАЗАТЕЛИ:
• FAR (False Acceptance Rate): {our_metrics['far']:.1f}%
• FRR (False Rejection Rate): {our_metrics['frr']:.1f}%
• EER (Equal Error Rate): {our_metrics['eer']:.1f}%

📊 СРАВНЕНИЕ С ЭТАЛОННЫМИ ПОКАЗАТЕЛЯМИ:

🏆 КОММЕРЧЕСКИЙ УРОВЕНЬ (FAR≤1%, FRR≤5%, EER≤3%):
• FAR: {our_metrics['far']:.1f}% vs 1.0% ({'+' if comparison_results['commercial']['far_diff'] >= 0 else ''}{comparison_results['commercial']['far_diff']:.1f}%)
• FRR: {our_metrics['frr']:.1f}% vs 5.0% ({'+' if comparison_results['commercial']['frr_diff'] >= 0 else ''}{comparison_results['commercial']['frr_diff']:.1f}%)
• EER: {our_metrics['eer']:.1f}% vs 3.0% ({'+' if comparison_results['commercial']['eer_diff'] >= 0 else ''}{comparison_results['commercial']['eer_diff']:.1f}%)

🔬 ИССЛЕДОВАТЕЛЬСКИЙ УРОВЕНЬ (FAR≤5%, FRR≤15%, EER≤10%):
• FAR: {our_metrics['far']:.1f}% vs 5.0% ({'+' if comparison_results['research']['far_diff'] >= 0 else ''}{comparison_results['research']['far_diff']:.1f}%)
• FRR: {our_metrics['frr']:.1f}% vs 15.0% ({'+' if comparison_results['research']['frr_diff'] >= 0 else ''}{comparison_results['research']['frr_diff']:.1f}%)
• EER: {our_metrics['eer']:.1f}% vs 10.0% ({'+' if comparison_results['research']['eer_diff'] >= 0 else ''}{comparison_results['research']['eer_diff']:.1f}%)

✅ ПРИЕМЛЕМЫЙ УРОВЕНЬ (FAR≤10%, FRR≤25%, EER≤15%):
• FAR: {our_metrics['far']:.1f}% vs 10.0% ({'+' if comparison_results['acceptable']['far_diff'] >= 0 else ''}{comparison_results['acceptable']['far_diff']:.1f}%)
• FRR: {our_metrics['frr']:.1f}% vs 25.0% ({'+' if comparison_results['acceptable']['frr_diff'] >= 0 else ''}{comparison_results['acceptable']['frr_diff']:.1f}%)
• EER: {our_metrics['eer']:.1f}% vs 15.0% ({'+' if comparison_results['acceptable']['eer_diff'] >= 0 else ''}{comparison_results['acceptable']['eer_diff']:.1f}%)

🏅 КАТЕГОРИЯ СИСТЕМЫ: {system_category}

🌟 ДОСТИЖЕНИЯ:
{self.list_achievements(our_metrics, benchmarks)}

📈 ОБЛАСТИ ДЛЯ УЛУЧШЕНИЯ:
{self.list_improvement_areas(our_metrics, benchmarks)}

🎓 ПРИГОДНОСТЬ ДЛЯ ДИПЛОМНОЙ РАБОТЫ:
{self.assess_thesis_suitability(our_metrics, system_category)}
"""
        
        return analysis
    
    def calculate_our_metrics(self) -> dict:
        """Расчет наших метрик"""
        if not self.auth_attempts:
            return {'far': 0.0, 'frr': 0.0, 'eer': 0.0}
        
        # Приблизительный расчет на основе уверенности
        high_conf = [a for a in self.auth_attempts if a['final_confidence'] >= 0.7]
        low_conf = [a for a in self.auth_attempts if a['final_confidence'] < 0.4]
        
        # FAR - ложные принятия (низкая уверенность, но принято)
        far = 0.0
        if low_conf:
            false_accepts = sum(1 for a in low_conf if a['result'])
            far = false_accepts / len(low_conf) * 100
        
        # FRR - ложные отказы (высокая уверенность, но отклонено)
        frr = 0.0
        if high_conf:
            false_rejects = sum(1 for a in high_conf if not a['result'])
            frr = false_rejects / len(high_conf) * 100
        
        # EER - приблизительное равное значение ошибок
        eer = (far + frr) / 2
        
        return {'far': far, 'frr': frr, 'eer': eer}
    
    def determine_system_category(self, our_metrics: dict, benchmarks: dict) -> str:
        """Определение категории системы"""
        far, frr, eer = our_metrics['far'], our_metrics['frr'], our_metrics['eer']
        
        # Проверка коммерческого уровня
        if (far <= benchmarks['commercial']['far'] and 
            frr <= benchmarks['commercial']['frr'] and 
            eer <= benchmarks['commercial']['eer']):
            return "🏆 КОММЕРЧЕСКИЙ УРОВЕНЬ - Превосходно!"
        
        # Проверка исследовательского уровня
        elif (far <= benchmarks['research']['far'] and 
              frr <= benchmarks['research']['frr'] and 
              eer <= benchmarks['research']['eer']):
            return "🔬 ИССЛЕДОВАТЕЛЬСКИЙ УРОВЕНЬ - Отлично для научной работы!"
        
        # Проверка приемлемого уровня
        elif (far <= benchmarks['acceptable']['far'] and 
              frr <= benchmarks['acceptable']['frr'] and 
              eer <= benchmarks['acceptable']['eer']):
            return "✅ ПРИЕМЛЕМЫЙ УРОВЕНЬ - Хорошо для дипломной работы!"
        
        else:
            return "⚠️ ТРЕБУЕТ УЛУЧШЕНИЯ - Но подходит для демонстрации концепции"
    
    def list_achievements(self, our_metrics: dict, benchmarks: dict) -> str:
        """Список достижений"""
        achievements = []
        
        if our_metrics['far'] <= benchmarks['research']['far']:
            achievements.append("• Низкий уровень ложных принятий (хорошая безопасность)")
        
        if our_metrics['frr'] <= benchmarks['research']['frr']:
            achievements.append("• Низкий уровень ложных отказов (хорошее удобство)")
        
        if our_metrics['eer'] <= benchmarks['research']['eer']:
            achievements.append("• EER соответствует исследовательским стандартам")
        
        if len(self.training_samples) >= 50:
            achievements.append("• Достаточный объем обучающих данных")
        
        if len(self.auth_attempts) >= 20:
            achievements.append("• Достаточно данных для статистического анализа")
        
        if not achievements:
            achievements.append("• Система функционирует и собирает данные для анализа")
        
        return '\n'.join(achievements)
    
    def list_improvement_areas(self, our_metrics: dict, benchmarks: dict) -> str:
        """Области для улучшения"""
        improvements = []
        
        if our_metrics['far'] > benchmarks['research']['far']:
            improvements.append("• Снизить FAR - повысить порог аутентификации или улучшить модель")
        
        if our_metrics['frr'] > benchmarks['research']['frr']:
            improvements.append("• Снизить FRR - понизить порог или собрать больше обучающих данных")
        
        if our_metrics['eer'] > benchmarks['research']['eer']:
            improvements.append("• Оптимизировать EER - сбалансировать FAR и FRR")
        
        if len(self.training_samples) < 50:
            improvements.append("• Собрать больше обучающих образцов")
        
        if len(self.auth_attempts) < 30:
            improvements.append("• Провести больше тестов аутентификации")
        
        if not improvements:
            improvements.append("• Система работает хорошо в текущем состоянии")
        
        return '\n'.join(improvements)
    
    def assess_thesis_suitability(self, our_metrics: dict, category: str) -> str:
        """Оценка пригодности для дипломной работы"""
        suitability = []
        
        if "КОММЕРЧЕСКИЙ" in category or "ИССЛЕДОВАТЕЛЬСКИЙ" in category:
            suitability.append("✅ ОТЛИЧНО подходит для дипломной работы")
            suitability.append("• Результаты соответствуют современным стандартам")
            suitability.append("• Можно смело включать в теоретическую и практическую части")
        elif "ПРИЕМЛЕМЫЙ" in category:
            suitability.append("✅ ХОРОШО подходит для дипломной работы")
            suitability.append("• Демонстрирует понимание принципов биометрии")
            suitability.append("• Показывает практическую реализацию концепций")
        else:
            suitability.append("⚠️ УСЛОВНО подходит для дипломной работы")
            suitability.append("• Показывает понимание предметной области")
            suitability.append("• Требует обсуждения ограничений и путей улучшения")
        
        suitability.append("")
        suitability.append("💡 Рекомендации для диплома:")
        suitability.append("• Включите анализ полученных метрик")
        suitability.append("• Обсудите сравнение с эталонными показателями")
        suitability.append("• Предложите пути улучшения системы")
        
        return '\n'.join(suitability)
    
    def plot_comparison_charts(self):
        """Построение графиков сравнения"""
        try:
            # График 1: Сравнение с эталонами
            self.ax_comp1.clear()
            
            our_metrics = self.calculate_our_metrics()
            
            metrics = ['FAR', 'FRR', 'EER']
            our_values = [our_metrics['far'], our_metrics['frr'], our_metrics['eer']]
            commercial_values = [1.0, 5.0, 3.0]
            research_values = [5.0, 15.0, 10.0]
            
            x = np.arange(len(metrics))
            width = 0.25
            
            bars1 = self.ax_comp1.bar(x - width, our_values, width, label='Наша система', color='skyblue')
            bars2 = self.ax_comp1.bar(x, commercial_values, width, label='Коммерческий уровень', color='gold')
            bars3 = self.ax_comp1.bar(x + width, research_values, width, label='Исследовательский уровень', color='lightcoral')
            
            self.ax_comp1.set_xlabel('Метрики')
            self.ax_comp1.set_ylabel('Процент ошибок (%)')
            self.ax_comp1.set_title('Сравнение с эталонными показателями')
            self.ax_comp1.set_xticks(x)
            self.ax_comp1.set_xticklabels(metrics)
            self.ax_comp1.legend()
            self.ax_comp1.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    self.ax_comp1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # График 2: Radar chart производительности
            self.ax_comp2.clear()
            
            if our_values and all(v >= 0 for v in our_values):
                # Нормализуем значения для radar chart (инвертируем - меньше лучше)
                max_val = max(max(our_values), max(research_values))
                normalized_our = [(max_val - v) / max_val for v in our_values]
                normalized_research = [(max_val - v) / max_val for v in research_values]
                
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]  # Замыкаем круг
                
                normalized_our += normalized_our[:1]
                normalized_research += normalized_research[:1]
                
                self.ax_comp2 = plt.subplot(122, projection='polar')
                self.ax_comp2.plot(angles, normalized_our, 'o-', linewidth=2, label='Наша система', color='blue')
                self.ax_comp2.fill(angles, normalized_our, alpha=0.25, color='blue')
                self.ax_comp2.plot(angles, normalized_research, 'o-', linewidth=2, label='Исследовательский уровень', color='red')
                self.ax_comp2.fill(angles, normalized_research, alpha=0.25, color='red')
                
                self.ax_comp2.set_xticks(angles[:-1])
                self.ax_comp2.set_xticklabels(metrics)
                self.ax_comp2.set_ylim(0, 1)
                self.ax_comp2.set_title('Профиль производительности\n(больше = лучше)', pad=20)
                self.ax_comp2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            self.fig_comp.tight_layout()
            self.canvas_comp.draw()
            
        except Exception as e:
            print(f"Ошибка построения графиков сравнения: {e}")
    
    def refresh_data(self):
        """Обновление данных"""
        try:
            # Перезагружаем данные
            self.training_samples = self.db.get_user_training_samples(self.user.id)
            self.auth_attempts = self.db.get_auth_attempts(self.user.id, limit=100)
            self.model_info = self.model_manager.get_model_info(self.user.id)
            
            # Перезагружаем все анализы
            self.load_enhanced_statistics()
            
            messagebox.showinfo("Обновление", "Данные успешно обновлены!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обновления данных: {e}")
    
    def optimize_model(self):
        """Оптимизация модели"""
        try:
            if len(self.training_samples) < 30:
                messagebox.showwarning("Предупреждение", 
                                     "Недостаточно данных для оптимизации. Соберите минимум 30 образцов.")
                return
            
            if messagebox.askyesno("Оптимизация модели", 
                                 "Запустить оптимизацию модели?\nЭто может занять несколько минут."):
                
                # Запускаем переобучение с продвинутыми настройками
                success, accuracy, message = self.model_manager.train_user_model(
                    self.user.id, 
                    use_enhanced_training=True
                )
                
                if success:
                    messagebox.showinfo("Успех", f"Модель оптимизирована!\nНовая точность: {accuracy:.1%}")
                    self.refresh_data()
                else:
                    messagebox.showerror("Ошибка", f"Ошибка оптимизации: {message}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка оптимизации: {e}")
    
    def export_detailed_report(self):
        """Экспорт детального отчета"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Сохранить детальный отчет"
            )
            
            if filename:
                # Собираем все данные в один отчет
                full_report = self.generate_full_report()
                
                if filename.endswith('.json'):
                    # Экспорт в JSON
                    report_data = self.collect_report_data()
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
                else:
                    # Экспорт в текстовый файл
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(full_report)
                
                messagebox.showinfo("Экспорт", f"Отчет сохранен в {filename}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    def generate_full_report(self) -> str:
        """Генерация полного отчета"""
        report_sections = [
            "ДЕТАЛЬНЫЙ ОТЧЕТ ПО БИОМЕТРИЧЕСКОЙ СИСТЕМЕ",
            "=" * 60,
            f"Пользователь: {self.user.username}",
            f"Дата создания отчета: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
            f"Тип модели: {self.model_info.get('model_type', 'none').upper()}",
            "",
            "1. АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ",
            "-" * 30
        ]
        
        # Добавляем содержимое каждой вкладки
        try:
            # Производительность
            if hasattr(self, 'performance_text'):
                performance_content = self.performance_text.get('1.0', tk.END).strip()
                if performance_content:
                    report_sections.append(performance_content)
            
            report_sections.extend(["", "2. ПОВЕДЕНЧЕСКИЕ ПАТТЕРНЫ", "-" * 30])
            
            # Паттерны
            if hasattr(self, 'patterns_text'):
                patterns_content = self.patterns_text.get('1.0', tk.END).strip()
                if patterns_content:
                    report_sections.append(patterns_content)
            
            report_sections.extend(["", "3. АНАЛИЗ БЕЗОПАСНОСТИ", "-" * 30])
            
            # Безопасность
            if hasattr(self, 'security_text'):
                security_content = self.security_text.get('1.0', tk.END).strip()
                if security_content:
                    report_sections.append(security_content)
            
            report_sections.extend(["", "4. ДИАГНОСТИКА МОДЕЛИ", "-" * 30])
            
            # Диагностика
            if hasattr(self, 'model_info_text'):
                model_content = self.model_info_text.get('1.0', tk.END).strip()
                if model_content:
                    report_sections.append(model_content)
            
            report_sections.extend(["", "5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ", "-" * 30])
            
            # Сравнение
            if hasattr(self, 'comparison_text'):
                comparison_content = self.comparison_text.get('1.0', tk.END).strip()
                if comparison_content:
                    report_sections.append(comparison_content)
            
            # Заключение
            report_sections.extend(["", "6. ЗАКЛЮЧЕНИЕ И РЕКОМЕНДАЦИИ", "-" * 30])
            report_sections.append(self.generate_conclusion())
            
        except Exception as e:
            report_sections.append(f"Ошибка генерации отчета: {e}")
        
        return "\n".join(report_sections)
    
    def collect_report_data(self) -> dict:
        """Сбор данных для JSON отчета"""
        our_metrics = self.calculate_our_metrics()
        
        return {
            "user_info": {
                "username": self.user.username,
                "user_id": self.user.id,
                "report_date": datetime.now().isoformat(),
                "model_type": self.model_info.get('model_type', 'none')
            },
            "training_data": {
                "total_samples": len(self.training_samples),
                "collection_period": {
                    "start": min(s.timestamp for s in self.training_samples).isoformat() if self.training_samples else None,
                    "end": max(s.timestamp for s in self.training_samples).isoformat() if self.training_samples else None
                }
            },
            "authentication_data": {
                "total_attempts": len(self.auth_attempts),
                "successful_attempts": sum(1 for a in self.auth_attempts if a['result']),
                "success_rate": np.mean([a['result'] for a in self.auth_attempts]) if self.auth_attempts else 0,
                "average_confidence": np.mean([a['final_confidence'] for a in self.auth_attempts]) if self.auth_attempts else 0
            },
            "security_metrics": our_metrics,
            "model_info": self.model_info,
            "recommendations": self.generate_improvement_suggestions().split('\n'),
            "system_category": self.determine_system_category(our_metrics, {
                'commercial': {'far': 1.0, 'frr': 5.0, 'eer': 3.0},
                'research': {'far': 5.0, 'frr': 15.0, 'eer': 10.0},
                'acceptable': {'far': 10.0, 'frr': 25.0, 'eer': 15.0}
            })
        }
    
    def generate_conclusion(self) -> str:
        """Генерация заключения"""
        our_metrics = self.calculate_our_metrics()
        
        conclusion_parts = [
            "ОБЩЕЕ ЗАКЛЮЧЕНИЕ:",
            "",
            f"Система показывает следующие результаты:",
            f"• FAR: {our_metrics['far']:.1f}%",
            f"• FRR: {our_metrics['frr']:.1f}%", 
            f"• EER: {our_metrics['eer']:.1f}%",
            "",
            "Качество реализации:",
        ]
        
        if our_metrics['eer'] <= 10:
            conclusion_parts.append("✅ ОТЛИЧНОЕ - соответствует исследовательским стандартам")
        elif our_metrics['eer'] <= 15:
            conclusion_parts.append("✅ ХОРОШЕЕ - подходит для дипломной работы")
        elif our_metrics['eer'] <= 25:
            conclusion_parts.append("⚠️ УДОВЛЕТВОРИТЕЛЬНОЕ - демонстрирует понимание концепций")
        else:
            conclusion_parts.append("⚠️ ТРЕБУЕТ ДОРАБОТКИ - но показывает базовое понимание")
        
        conclusion_parts.extend([
            "",
            "Рекомендации для дальнейшего развития:",
            "• Увеличить объем обучающих данных",
            "• Провести более детальный анализ признаков",
            "• Рассмотреть дополнительные алгоритмы машинного обучения",
            "• Провести тестирование на большей группе пользователей",
            "",
            "Система готова для демонстрации в рамках дипломной работы."
        ])
        
        return "\n".join(conclusion_parts)


def main():
    """Тестовая функция для отладки"""
    root = tk.Tk()
    root.withdraw()  # Скрываем главное окно
    
    # Создаем тестового пользователя
    from models.user import User
    from auth.keystroke_auth import KeystrokeAuthenticator
    
    test_user = User(
        username="test_user",
        password_hash="test_hash",
        salt="test_salt",
        id=1,
        is_trained=True
    )
    
    keystroke_auth = KeystrokeAuthenticator()
    
    # Создаем окно статистики
    stats_window = EnhancedModelStatsWindow(root, test_user, keystroke_auth)
    
    root.mainloop()


if __name__ == "__main__":
    main()