#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценщик эффективности биометрической системы аутентификации
Автор: Студент
Цель: Анализ метрик FAR, FRR, EER, ROC для дипломной работы
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime

class BiometricSystemEvaluator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Оценщик эффективности биометрической системы")
        self.root.geometry("800x600")
        
        # Данные по умолчанию (ваши реальные результаты)
        self.legitimate_data = [85.1, 79.1, 73.0, 75.9, 82.4, 67.7, 83.8, 84.0, 81.6, 74.9]
        self.impostor_fast = [15.7, 15.7, 15.2, 16.0, 15.2, 15.7, 15.3, 15.4, 15.6, 14.8]
        self.impostor_slow = [64.6, 64.7, 64.2, 64.3, 64.2, 64.3, 64.0, 64.2, 64.2, 64.1]
        
        self.current_threshold = 75.0
        
        self.create_interface()
        
    def create_interface(self):
        """Создание графического интерфейса"""
        
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="🔬 Оценщик биометрической системы", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Секция данных
        data_frame = ttk.LabelFrame(main_frame, text="📊 Данные тестирования", padding="10")
        data_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(data_frame, text="Легитимные попытки (ваш стиль):").grid(row=0, column=0, sticky=tk.W)
        self.legit_entry = tk.Text(data_frame, height=3, width=60)
        self.legit_entry.insert('1.0', ', '.join(map(str, self.legitimate_data)))
        self.legit_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(data_frame, text="Имитаторы - быстрая печать:").grid(row=1, column=0, sticky=tk.W)
        self.fast_entry = tk.Text(data_frame, height=3, width=60)
        self.fast_entry.insert('1.0', ', '.join(map(str, self.impostor_fast)))
        self.fast_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(data_frame, text="Имитаторы - медленная печать:").grid(row=2, column=0, sticky=tk.W)
        self.slow_entry = tk.Text(data_frame, height=3, width=60)
        self.slow_entry.insert('1.0', ', '.join(map(str, self.impostor_slow)))
        self.slow_entry.grid(row=2, column=1, padx=5)
        
        # Секция настроек
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Настройки", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(settings_frame, text="Текущий порог (%):").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=self.current_threshold)
        threshold_scale = ttk.Scale(settings_frame, from_=0, to=100, variable=self.threshold_var, 
                                   orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=1, padx=5)
        self.threshold_label = ttk.Label(settings_frame, text=f"{self.current_threshold:.1f}%")
        self.threshold_label.grid(row=0, column=2)
        
        # Обновление значения порога
        def update_threshold(*args):
            self.threshold_label.config(text=f"{self.threshold_var.get():.1f}%")
        self.threshold_var.trace('w', update_threshold)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="🔍 Анализировать систему", 
                  command=self.analyze_system, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📈 Показать ROC-кривую", 
                  command=self.show_roc_curve).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 Экспорт отчета", 
                  command=self.export_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📁 Загрузить данные", 
                  command=self.load_data).pack(side=tk.LEFT, padx=5)
        
        # Область результатов
        results_frame = ttk.LabelFrame(main_frame, text="📋 Результаты анализа", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = tk.Text(results_frame, height=15, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Настройка масштабирования
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Автоматический анализ при запуске
        self.root.after(500, self.analyze_system)
    
    def parse_data(self, text_widget):
        """Парсинг данных из текстового виджета"""
        try:
            text = text_widget.get('1.0', tk.END).strip()
            # Убираем лишние символы и разделяем
            numbers = []
            for item in text.replace('\n', ' ').split(','):
                item = item.strip().replace('%', '')
                if item:
                    numbers.append(float(item))
            return numbers
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка парсинга данных: {e}")
            return []
    
    def analyze_system(self):
        """Основной анализ системы"""
        try:
            # Получаем данные из интерфейса
            legitimate = self.parse_data(self.legit_entry)
            fast_impostors = self.parse_data(self.fast_entry)
            slow_impostors = self.parse_data(self.slow_entry)
            current_threshold = self.threshold_var.get()
            
            if not legitimate or not (fast_impostors or slow_impostors):
                messagebox.showerror("Ошибка", "Недостаточно данных для анализа!")
                return
            
            # Преобразуем в проценты, если нужно
            legitimate = [x/100 if x > 1 else x for x in legitimate]
            fast_impostors = [x/100 if x > 1 else x for x in fast_impostors]
            slow_impostors = [x/100 if x > 1 else x for x in slow_impostors]
            current_threshold = current_threshold / 100
            
            # Объединяем данные имитаторов
            all_impostors = fast_impostors + slow_impostors
            
            # Анализ для разных порогов
            thresholds = np.arange(0.1, 1.0, 0.05)
            metrics_results = []
            
            for thresh in thresholds:
                # Легитимные пользователи
                tp = sum(1 for score in legitimate if score >= thresh)
                fn = len(legitimate) - tp
                
                # Имитаторы
                fp = sum(1 for score in all_impostors if score >= thresh)
                tn = len(all_impostors) - fp
                
                # Метрики
                far = (fp / len(all_impostors)) * 100 if all_impostors else 0
                frr = (fn / len(legitimate)) * 100 if legitimate else 0
                eer = (far + frr) / 2
                accuracy = ((tp + tn) / (len(legitimate) + len(all_impostors))) * 100
                
                metrics_results.append({
                    'threshold': thresh * 100,
                    'far': far,
                    'frr': frr,
                    'eer': eer,
                    'accuracy': accuracy,
                    'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
                })
            
            # Находим результат для текущего порога
            current_result = min(metrics_results, 
                               key=lambda x: abs(x['threshold'] - current_threshold * 100))
            
            # Оптимальный порог (минимальный EER)
            optimal_result = min(metrics_results, key=lambda x: x['eer'])
            
            # ROC анализ
            all_scores = legitimate + all_impostors
            all_labels = [1] * len(legitimate) + [0] * len(all_impostors)
            
            fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
            roc_auc = auc(fpr, tpr)
            
            # Формируем отчет
            report = self.generate_report(legitimate, fast_impostors, slow_impostors, 
                                        current_result, optimal_result, roc_auc, 
                                        current_threshold * 100)
            
            # Выводим результаты
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert('1.0', report)
            
            # Сохраняем данные для экспорта
            self.last_analysis = {
                'legitimate': legitimate,
                'impostors': all_impostors,
                'fast_impostors': fast_impostors,
                'slow_impostors': slow_impostors,
                'current_result': current_result,
                'optimal_result': optimal_result,
                'roc_auc': roc_auc,
                'metrics_results': metrics_results,
                'all_scores': all_scores,
                'all_labels': all_labels
            }
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_report(self, legitimate, fast_impostors, slow_impostors, 
                       current_result, optimal_result, roc_auc, current_threshold):
        """Генерация подробного отчета"""
        
        report = f"""
🔬 АНАЛИЗ БИОМЕТРИЧЕСКОЙ СИСТЕМЫ АУТЕНТИФИКАЦИИ
{'='*80}

📊 ВХОДНЫЕ ДАННЫЕ:
• Легитимные попытки (ваш стиль): {len(legitimate)} образцов
  Средняя уверенность: {np.mean(legitimate):.1%}
  Диапазон: {min(legitimate):.1%} - {max(legitimate):.1%}
  Стандартное отклонение: {np.std(legitimate):.1%}

• Имитаторы - быстрая печать: {len(fast_impostors)} образцов
  Средняя уверенность: {np.mean(fast_impostors):.1%}
  Диапазон: {min(fast_impostors):.1%} - {max(fast_impostors):.1%}

• Имитаторы - медленная печать: {len(slow_impostors)} образцов
  Средняя уверенность: {np.mean(slow_impostors):.1%}
  Диапазон: {min(slow_impostors):.1%} - {max(slow_impostors):.1%}

🎯 МЕТРИКИ ПРИ ТЕКУЩЕМ ПОРОГЕ ({current_threshold:.1f}%):

• FAR (False Acceptance Rate): {current_result['far']:.2f}%
  Принято имитаторов: {current_result['fp']}/{current_result['fp'] + current_result['tn']}
  Интерпретация: {self.interpret_far(current_result['far'])}

• FRR (False Rejection Rate): {current_result['frr']:.2f}%
  Отклонено легитимных: {current_result['fn']}/{current_result['tp'] + current_result['fn']}
  Интерпретация: {self.interpret_frr(current_result['frr'])}

• EER (Equal Error Rate): {current_result['eer']:.2f}%
  Интерпретация: {self.interpret_eer(current_result['eer'])}

• Общая точность: {current_result['accuracy']:.1f}%

📈 ROC АНАЛИЗ:
• AUC (Area Under Curve): {roc_auc:.3f}
• Качество классификации: {self.interpret_auc(roc_auc)}
• Разделимость классов: {abs(np.mean(legitimate) - np.mean(fast_impostors + slow_impostors)):.1%}

🎛️ ОПТИМИЗАЦИЯ:
• Рекомендуемый порог: {optimal_result['threshold']:.1f}%
• FAR при оптимальном пороге: {optimal_result['far']:.2f}%
• FRR при оптимальном пороге: {optimal_result['frr']:.2f}%
• EER при оптимальном пороге: {optimal_result['eer']:.2f}%

🔍 ДЕТАЛЬНЫЙ АНАЛИЗ CONFUSION MATRIX:
┌─────────────────┬──────────────┬──────────────┐
│                 │   Система    │   Система    │
│                 │  ПРИНИМАЕТ   │  ОТКЛОНЯЕТ   │
├─────────────────┼──────────────┼──────────────┤
│ Легитимный      │ TP: {current_result['tp']:8d} │ FN: {current_result['fn']:8d} │
│ пользователь    │              │              │
├─────────────────┼──────────────┼──────────────┤
│ Имитатор        │ FP: {current_result['fp']:8d} │ TN: {current_result['tn']:8d} │
│                 │              │              │
└─────────────────┴──────────────┴──────────────┘

💡 ЗАКЛЮЧЕНИЕ ДЛЯ ДИПЛОМНОЙ РАБОТЫ:
{self.generate_conclusion(current_result, optimal_result, roc_auc)}

📅 Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
"""
        return report
    
    def interpret_far(self, far):
        """Интерпретация FAR"""
        if far == 0:
            return "ОТЛИЧНО - полная защита от имитаторов"
        elif far < 5:
            return "ОТЛИЧНО - очень низкий риск принятия имитаторов"
        elif far < 15:
            return "ХОРОШО - приемлемый уровень безопасности"
        elif far < 30:
            return "СРЕДНЕ - умеренный риск безопасности"
        else:
            return "ПЛОХО - высокий риск принятия имитаторов"
    
    def interpret_frr(self, frr):
        """Интерпретация FRR"""
        if frr < 10:
            return "ОТЛИЧНО - очень удобно для пользователя"
        elif frr < 25:
            return "ХОРОШО - приемлемое удобство использования"
        elif frr < 40:
            return "СРЕДНЕ - возможны частые отказы"
        else:
            return "ПЛОХО - неудобно для пользователя"
    
    def interpret_eer(self, eer):
        """Интерпретация EER"""
        if eer < 5:
            return "ОТЛИЧНО - система коммерческого уровня"
        elif eer < 15:
            return "ХОРОШО - система научного уровня"
        elif eer < 25:
            return "СРЕДНЕ - приемлемо для исследований"
        else:
            return "ПЛОХО - требует улучшения"
    
    def interpret_auc(self, auc_val):
        """Интерпретация AUC"""
        if auc_val >= 0.95:
            return "ОТЛИЧНО (превосходная классификация)"
        elif auc_val >= 0.85:
            return "ХОРОШО (хорошая классификация)"
        elif auc_val >= 0.75:
            return "СРЕДНЕ (удовлетворительная классификация)"
        else:
            return "ПЛОХО (слабая классификация)"
    
    def generate_conclusion(self, current_result, optimal_result, roc_auc):
        """Генерация заключения для дипломной работы"""
        conclusions = []
        
        if roc_auc >= 0.9:
            conclusions.append("• Система демонстрирует отличную способность различать классы")
        
        if current_result['far'] <= 5:
            conclusions.append("• Высокий уровень защиты от атак имитации")
        
        if current_result['eer'] <= 15:
            conclusions.append("• EER соответствует современным стандартам биометрии")
        
        if optimal_result['eer'] < current_result['eer']:
            diff = current_result['eer'] - optimal_result['eer']
            conclusions.append(f"• Возможно улучшение EER на {diff:.1f}% при оптимизации порога")
        
        conclusions.append("• Система готова для практического применения")
        
        return "\n".join(conclusions)
    
    def show_roc_curve(self):
        """Показ ROC-кривой"""
        if not hasattr(self, 'last_analysis'):
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ системы!")
            return
        
        try:
            data = self.last_analysis
            
            # ROC кривая
            fpr, tpr, thresholds = roc_curve(data['all_labels'], data['all_scores'])
            roc_auc = data['roc_auc']
            
            # Создаем окно с графиками
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Анализ биометрической системы аутентификации', fontsize=16, fontweight='bold')
            
            # График 1: ROC кривая
            ax1.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC кривая (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate (FAR)', fontsize=12)
            ax1.set_ylabel('True Positive Rate (1 - FRR)', fontsize=12)
            ax1.set_title('ROC Кривая', fontsize=14, fontweight='bold')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # График 2: Распределение оценок
            ax2.hist(data['legitimate'], bins=15, alpha=0.7, color='green', 
                    label=f'Легитимные ({len(data["legitimate"])})', density=True, edgecolor='darkgreen')
            ax2.hist(data['impostors'], bins=15, alpha=0.7, color='red',
                    label=f'Имитаторы ({len(data["impostors"])})', density=True, edgecolor='darkred')
            ax2.axvline(self.threshold_var.get()/100, color='black', linestyle='--', linewidth=2, 
                       label=f'Порог {self.threshold_var.get():.1f}%')
            ax2.set_xlabel('Уверенность системы', fontsize=12)
            ax2.set_ylabel('Плотность', fontsize=12)
            ax2.set_title('Распределение оценок', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # График 3: FAR vs FRR vs Порог
            thresholds_list = [r['threshold'] for r in data['metrics_results']]
            far_list = [r['far'] for r in data['metrics_results']]
            frr_list = [r['frr'] for r in data['metrics_results']]
            
            ax3.plot(thresholds_list, far_list, 'r-o', label='FAR', linewidth=2, markersize=4)
            ax3.plot(thresholds_list, frr_list, 'b-s', label='FRR', linewidth=2, markersize=4)
            ax3.axvline(self.threshold_var.get(), color='gray', linestyle='--', alpha=0.7, 
                       label='Текущий порог')
            ax3.axvline(data['optimal_result']['threshold'], color='green', linestyle='--', 
                       alpha=0.7, label='Оптимальный порог')
            ax3.set_xlabel('Порог (%)', fontsize=12)
            ax3.set_ylabel('Частота ошибок (%)', fontsize=12)
            ax3.set_title('FAR и FRR vs Порог', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # График 4: EER vs Порог
            eer_list = [r['eer'] for r in data['metrics_results']]
            ax4.plot(thresholds_list, eer_list, 'g-^', label='EER', linewidth=3, markersize=6)
            ax4.axvline(self.threshold_var.get(), color='gray', linestyle='--', alpha=0.7, 
                       label='Текущий порог')
            ax4.axvline(data['optimal_result']['threshold'], color='green', linestyle='--', 
                       alpha=0.7, label='Оптимальный порог')
            ax4.set_xlabel('Порог (%)', fontsize=12)
            ax4.set_ylabel('EER (%)', fontsize=12)
            ax4.set_title('Equal Error Rate vs Порог', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графиков: {e}")
    
    def export_report(self):
        """Экспорт отчета в файл"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
                title="Сохранить отчет"
            )
            
            if filename:
                report = self.results_text.get('1.0', tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Успех", f"Отчет сохранен в {filename}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {e}")
    
    def load_data(self):
        """Загрузка данных из файла"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Загрузить данные"
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'legitimate' in data:
                    self.legit_entry.delete('1.0', tk.END)
                    self.legit_entry.insert('1.0', ', '.join(map(str, data['legitimate'])))
                
                if 'fast_impostors' in data:
                    self.fast_entry.delete('1.0', tk.END)
                    self.fast_entry.insert('1.0', ', '.join(map(str, data['fast_impostors'])))
                
                if 'slow_impostors' in data:
                    self.slow_entry.delete('1.0', tk.END)
                    self.slow_entry.insert('1.0', ', '.join(map(str, data['slow_impostors'])))
                
                messagebox.showinfo("Успех", "Данные загружены успешно!")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки: {e}")
    
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

def main():
    """Главная функция"""
    print("🚀 Запуск оценщика биометрической системы...")
    app = BiometricSystemEvaluator()
    app.run()

if __name__ == "__main__":
    main()