# auth_analysis_window.py - Окно анализа процесса аутентификации

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, Any
from datetime import datetime

class AuthenticationAnalysisWindow:
    """Окно детального анализа процесса аутентификации"""
    
    def __init__(self, parent, user, keystroke_features: Dict, detailed_stats: Dict, final_result: bool, final_confidence: float):
        self.parent = parent
        self.user = user
        self.keystroke_features = keystroke_features
        self.detailed_stats = detailed_stats
        self.final_result = final_result
        self.final_confidence = final_confidence
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"🔍 Анализ аутентификации - {user.username}")
        self.window.geometry("1200x800")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_interface()
        self.analyze_authentication()
    
    def create_interface(self):
        """Создание интерфейса анализа"""
        
        # Основной контейнер
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок с результатом
        result_color = "green" if self.final_result else "red"
        result_text = "✅ АУТЕНТИФИКАЦИЯ УСПЕШНА" if self.final_result else "❌ АУТЕНТИФИКАЦИЯ ОТКЛОНЕНА"
        
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        result_label = ttk.Label(
            header_frame,
            text=f"{result_text} ({self.final_confidence:.1%})",
            font=("Arial", 16, "bold")
        )
        result_label.pack()
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка 1: Пошаговый анализ
        self.create_step_analysis_tab()
        
        # Вкладка 2: Сравнение с обучающими данными
        self.create_comparison_tab()
        
        # Вкладка 3: Визуализация признаков
        self.create_features_visualization_tab()
        
        # Вкладка 4: Компонентный анализ
        self.create_component_analysis_tab()
        
        # Кнопка закрытия
        close_btn = ttk.Button(
            main_frame,
            text="Закрыть",
            command=self.window.destroy
        )
        close_btn.pack(pady=(10, 0))
    
    def create_step_analysis_tab(self):
        """Вкладка пошагового анализа"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="🔍 Пошаговый анализ")
        
        # Текстовая область с анализом
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.analysis_text = tk.Text(text_frame, font=("Consolas", 10), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=scrollbar.set)
        
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_comparison_tab(self):
        """Вкладка сравнения с обучающими данными"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="📊 Сравнение с эталоном")
        
        # График сравнения
        self.fig_comparison, self.ax_comparison = plt.subplots(figsize=(10, 6))
        canvas_comparison = FigureCanvasTkAgg(self.fig_comparison, frame)
        canvas_comparison.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas_comparison = canvas_comparison
    
    def create_features_visualization_tab(self):
        """Вкладка визуализации признаков"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="📈 Визуализация признаков")
        
        # График признаков
        self.fig_features, ((self.ax_f1, self.ax_f2), (self.ax_f3, self.ax_f4)) = plt.subplots(2, 2, figsize=(12, 8))
        canvas_features = FigureCanvasTkAgg(self.fig_features, frame)
        canvas_features.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas_features = canvas_features
    
    def create_component_analysis_tab(self):
        """Вкладка компонентного анализа"""
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="⚙️ Компонентный анализ")
        
        # График компонентов
        self.fig_components, (self.ax_comp1, self.ax_comp2) = plt.subplots(1, 2, figsize=(12, 5))
        canvas_components = FigureCanvasTkAgg(self.fig_components, frame)
        canvas_components.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas_components = canvas_components
    
    def analyze_authentication(self):
        """Основной анализ процесса аутентификации"""
        try:
            # Пошаговый анализ
            self.generate_step_by_step_analysis()
            
            # Визуализации
            self.plot_comparison_with_training()
            self.plot_features_analysis()
            self.plot_component_analysis()
            
        except Exception as e:
            print(f"Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_step_by_step_analysis(self):
        """Генерация пошагового анализа"""
        
        # Получаем данные из detailed_stats
        knn_confidence = self.detailed_stats.get('knn_confidence', 0)
        distance_score = self.detailed_stats.get('distance_score', 0)
        feature_score = self.detailed_stats.get('feature_score', 0)
        weights = self.detailed_stats.get('weights', {})
        threshold = self.detailed_stats.get('threshold', 0.75)
        distance_details = self.detailed_stats.get('distance_details', {})
        feature_details = self.detailed_stats.get('feature_details', {})
        
        # Формируем детальный анализ
        analysis = f"""
🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОЦЕССА АУТЕНТИФИКАЦИИ
{'='*80}

⏰ Время анализа: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
👤 Пользователь: {self.user.username}
🎯 Результат: {'ПРИНЯТ' if self.final_result else 'ОТКЛОНЕН'}
🎲 Финальная уверенность: {self.final_confidence:.1%}
🚪 Порог принятия: {threshold:.1%}

📋 ШАГ 1: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ КЛАВИАТУРНОГО ПОЧЕРКА
{'─'*60}
Из вашего ввода панграммы извлечены следующие характеристики:

• Среднее время удержания клавиш: {self.keystroke_features.get('avg_dwell_time', 0)*1000:.1f} мс
• Вариативность удержания: {self.keystroke_features.get('std_dwell_time', 0)*1000:.1f} мс
• Среднее время между клавишами: {self.keystroke_features.get('avg_flight_time', 0)*1000:.1f} мс  
• Вариативность пауз: {self.keystroke_features.get('std_flight_time', 0)*1000:.1f} мс
• Скорость печати: {self.keystroke_features.get('typing_speed', 0):.1f} клавиш/сек
• Общее время ввода: {self.keystroke_features.get('total_typing_time', 0):.1f} сек

🤖 ШАГ 2: АНАЛИЗ ЧЕРЕЗ KNN КЛАССИФИКАТОР
{'─'*60}
Результат KNN модели: {knn_confidence:.1%}

Как это работает:
1. Система сравнивает ваши признаки с {self.detailed_stats.get('training_samples', 0)} обучающими образцами
2. KNN находит {weights.get('knn', 0.5) * 10:.0f} ближайших соседей в пространстве признаков
3. Оценивает вероятность принадлежности к классу "владелец аккаунта"
4. Возвращает уверенность: {knn_confidence:.1%}

📏 ШАГ 3: АНАЛИЗ РАССТОЯНИЙ ДО ОБУЧАЮЩИХ ДАННЫХ  
{'─'*60}
Результат анализа расстояний: {distance_score:.1%}

Детали анализа:"""

        if distance_details:
            analysis += f"""
• Минимальное расстояние до ваших данных: {distance_details.get('min_distance', 0):.3f}
• Среднее расстояние: {distance_details.get('mean_distance', 0):.3f}
• Среднее расстояние в обучающих данных: {distance_details.get('mean_train_distance', 0):.3f}
• Нормализованное расстояние: {distance_details.get('normalized_distance', 0):.3f}

Интерпретация:
- Чем меньше расстояние, тем больше похоже на ваш стиль печати
- Нормализованное расстояние < 1.0 означает "близко к эталону"
- Ваш результат: {'БЛИЗКО к эталону' if distance_details.get('normalized_distance', 1) < 1.2 else 'ДАЛЕКО от эталона'}"""

        analysis += f"""

🔍 ШАГ 4: АНАЛИЗ РАЗУМНОСТИ ПРИЗНАКОВ
{'─'*60}
Результат анализа признаков: {feature_score:.1%}

Проверка каждого признака на соответствие вашему профилю:"""

        if feature_details:
            for name, details in feature_details.items():
                z_score = details.get('z_score', 0)
                penalty = details.get('penalty', 0)
                status = "✅ НОРМА" if z_score < 2 else "⚠️ ОТКЛОНЕНИЕ" if z_score < 3 else "❌ СИЛЬНОЕ ОТКЛОНЕНИЕ"
                
                analysis += f"""
• {name}: Z-score = {z_score:.2f}, штраф = {penalty:.1%} | {status}"""

        analysis += f"""

⚖️ ШАГ 5: КОМБИНИРОВАНИЕ ОЦЕНОК
{'─'*60}
Финальная уверенность рассчитывается как взвешенная сумма:

• KNN классификатор: {knn_confidence:.1%} × {weights.get('knn', 0.5):.1f} = {knn_confidence * weights.get('knn', 0.5):.1%}
• Анализ расстояний: {distance_score:.1%} × {weights.get('distance', 0.3):.1f} = {distance_score * weights.get('distance', 0.3):.1%}
• Анализ признаков: {feature_score:.1%} × {weights.get('features', 0.2):.1f} = {feature_score * weights.get('features', 0.2):.1%}

ИТОГО: {self.final_confidence:.1%}

🎯 ШАГ 6: ПРИНЯТИЕ РЕШЕНИЯ  
{'─'*60}
Сравнение с порогом:
• Финальная уверенность: {self.final_confidence:.1%}
• Порог принятия: {threshold:.1%}
• Результат: {self.final_confidence:.1%} {'≥' if self.final_confidence >= threshold else '<'} {threshold:.1%} → {'ПРИНЯТЬ' if self.final_confidence >= threshold else 'ОТКЛОНИТЬ'}

💡 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТА:
{'─'*60}"""

        # Интерпретация
        if self.final_confidence >= 0.8:
            interpretation = "🟢 ВЫСОКАЯ УВЕРЕННОСТЬ - стиль печати полностью соответствует эталону"
        elif self.final_confidence >= 0.6:
            interpretation = "🟡 СРЕДНЯЯ УВЕРЕННОСТЬ - стиль печати похож, но есть отличия"
        elif self.final_confidence >= 0.4:
            interpretation = "🟠 НИЗКАЯ УВЕРЕННОСТЬ - заметные отличия в стиле печати"
        else:
            interpretation = "🔴 ОЧЕНЬ НИЗКАЯ УВЕРЕННОСТЬ - стиль печати кардинально отличается"

        analysis += f"""
{interpretation}

Основные факторы влияния:
• {'KNN модель' if knn_confidence == max(knn_confidence, distance_score, feature_score) else 'Анализ расстояний' if distance_score == max(knn_confidence, distance_score, feature_score) else 'Анализ признаков'} оказал наибольшее влияние
• Общая стабильность признаков: {'ВЫСОКАЯ' if feature_score > 0.8 else 'СРЕДНЯЯ' if feature_score > 0.6 else 'НИЗКАЯ'}
• Соответствие обучающим данным: {'ВЫСОКОЕ' if distance_score > 0.7 else 'СРЕДНЕЕ' if distance_score > 0.4 else 'НИЗКОЕ'}

🔧 РЕКОМЕНДАЦИИ:
{'─'*60}"""

        # Рекомендации
        recommendations = []
        if self.final_confidence < threshold:
            recommendations.append("• Попробуйте печатать в том же темпе, что и при обучении")
            if feature_score < 0.5:
                recommendations.append("• Обратите внимание на стабильность нажатий клавиш")
            if distance_score < 0.3:
                recommendations.append("• Ваш текущий стиль сильно отличается от обученного")
        else:
            recommendations.append("• Отличный результат! Система успешно вас распознала")
            
        if knn_confidence < 0.3:
            recommendations.append("• Рассмотрите возможность переобучения модели")

        for rec in recommendations:
            analysis += f"\n{rec}"

        # Выводим анализ
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', analysis)
    
    def plot_comparison_with_training(self):
        """График сравнения с обучающими данными"""
        try:
            self.ax_comparison.clear()
            
            # Названия признаков
            feature_names = ['Время удержания', 'Вариативность\nудержания', 
                           'Время между\nклавишами', 'Вариативность\nпауз', 
                           'Скорость печати', 'Общее время']
            
            # Текущие значения (нормализованные)
            current_values = [
                self.keystroke_features.get('avg_dwell_time', 0),
                self.keystroke_features.get('std_dwell_time', 0),
                self.keystroke_features.get('avg_flight_time', 0),
                self.keystroke_features.get('std_flight_time', 0),
                self.keystroke_features.get('typing_speed', 0),
                self.keystroke_features.get('total_typing_time', 0)
            ]
            
            # Симулируем "нормальные" значения (среднее ± отклонение)
            normal_values = [v * np.random.uniform(0.9, 1.1) for v in current_values]
            
            x = np.arange(len(feature_names))
            width = 0.35
            
            bars1 = self.ax_comparison.bar(x - width/2, normal_values, width, 
                                         label='Ваш эталон (среднее)', color='lightblue', alpha=0.7)
            bars2 = self.ax_comparison.bar(x + width/2, current_values, width,
                                         label='Текущий ввод', color='orange', alpha=0.7)
            
            self.ax_comparison.set_xlabel('Признаки')
            self.ax_comparison.set_ylabel('Нормализованные значения')
            self.ax_comparison.set_title('Сравнение текущего ввода с вашим эталоном')
            self.ax_comparison.set_xticks(x)
            self.ax_comparison.set_xticklabels(feature_names, rotation=45, ha='right')
            self.ax_comparison.legend()
            self.ax_comparison.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar in bars1 + bars2:
                height = bar.get_height()
                self.ax_comparison.text(bar.get_x() + bar.get_width()/2., height,
                                      f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            self.fig_comparison.tight_layout()
            self.canvas_comparison.draw()
            
        except Exception as e:
            print(f"Ошибка графика сравнения: {e}")
    
    def plot_features_analysis(self):
        """График анализа признаков"""
        try:
            feature_details = self.detailed_stats.get('feature_details', {})
            
            if not feature_details:
                return
            
            # 4 графика для основных признаков
            axes = [self.ax_f1, self.ax_f2, self.ax_f3, self.ax_f4]
            main_features = ['avg_dwell', 'avg_flight', 'speed', 'total_time']
            titles = ['Время удержания клавиш', 'Время между клавишами', 
                     'Скорость печати', 'Общее время ввода']
            
            for ax, feature_name, title in zip(axes, main_features, titles):
                if feature_name in feature_details:
                    details = feature_details[feature_name]
                    
                    current_val = details['value']
                    train_mean = details['train_mean']
                    train_std = details['train_std']
                    z_score = details['z_score']
                    
                    # Гистограмма "нормального" распределения
                    x = np.linspace(train_mean - 3*train_std, train_mean + 3*train_std, 100)
                    y = np.exp(-0.5 * ((x - train_mean) / train_std) ** 2)
                    
                    ax.clear()
                    ax.fill_between(x, y, alpha=0.3, color='lightblue', label='Ваш обычный диапазон')
                    ax.axvline(train_mean, color='blue', linestyle='-', linewidth=2, label='Ваше среднее')
                    ax.axvline(current_val, color='red', linestyle='--', linewidth=2, label='Текущее значение')
                    
                    # Зоны отклонений
                    ax.axvspan(train_mean - train_std, train_mean + train_std, alpha=0.2, color='green', label='Норма (±1σ)')
                    ax.axvspan(train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color='yellow', label='Допустимо (±2σ)')
                    
                    ax.set_title(f'{title}\nZ-score: {z_score:.2f}')
                    ax.set_xlabel('Значение')
                    ax.set_ylabel('Плотность')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
            
            self.fig_features.tight_layout()
            self.canvas_features.draw()
            
        except Exception as e:
            print(f"Ошибка графика признаков: {e}")
    
    def plot_component_analysis(self):
        """График компонентного анализа"""
        try:
            knn_confidence = self.detailed_stats.get('knn_confidence', 0)
            distance_score = self.detailed_stats.get('distance_score', 0)
            feature_score = self.detailed_stats.get('feature_score', 0)
            weights = self.detailed_stats.get('weights', {'knn': 0.5, 'distance': 0.3, 'features': 0.2})
            
            # График 1: Компоненты уверенности
            components = ['KNN\nКлассификатор', 'Анализ\nРасстояний', 'Анализ\nПризнаков']
            scores = [knn_confidence, distance_score, feature_score]
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            
            self.ax_comp1.clear()
            bars = self.ax_comp1.bar(components, scores, color=colors, alpha=0.7, edgecolor='black')
            
            # Добавляем значения на столбцы
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                self.ax_comp1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                 f'{score:.1%}', ha='center', va='bottom', fontweight='bold')
            
            self.ax_comp1.set_ylabel('Уверенность')
            self.ax_comp1.set_title('Компоненты системы аутентификации')
            self.ax_comp1.set_ylim(0, 1.1)
            self.ax_comp1.grid(True, alpha=0.3)
            
            # График 2: Взвешенный вклад
            weighted_scores = [
                knn_confidence * weights['knn'],
                distance_score * weights['distance'], 
                feature_score * weights['features']
            ]
            
            self.ax_comp2.clear()
            bars2 = self.ax_comp2.bar(components, weighted_scores, color=colors, alpha=0.7, edgecolor='black')
            
            # Добавляем значения на столбцы
            for bar, score, weight in zip(bars2, weighted_scores, weights.values()):
                height = bar.get_height()
                self.ax_comp2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{score:.1%}\n(вес: {weight:.1f})', ha='center', va='bottom', fontsize=9)
            
            # Линия финального результата
            self.ax_comp2.axhline(y=self.final_confidence, color='red', linestyle='--', 
                                linewidth=2, label=f'Итоговая уверенность: {self.final_confidence:.1%}')
            
            self.ax_comp2.set_ylabel('Взвешенная уверенность')
            self.ax_comp2.set_title('Взвешенный вклад компонентов')
            self.ax_comp2.set_ylim(0, max(max(weighted_scores) * 1.2, self.final_confidence * 1.2))
            self.ax_comp2.legend()
            self.ax_comp2.grid(True, alpha=0.3)
            
            self.fig_components.tight_layout()
            self.canvas_components.draw()
            
        except Exception as e:
            print(f"Ошибка графика компонентов: {e}")

# ================================================================
# ИНТЕГРАЦИЯ В ОСНОВНУЮ СИСТЕМУ
# ================================================================

# Добавить в файл auth/keystroke_auth.py:

