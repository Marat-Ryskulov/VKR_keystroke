# config.py - Обновленная конфигурация с адаптивными размерами

import os
import tkinter as tk

# Определяем размер экрана
def get_screen_info():
    """Получение информации о экране"""
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        return 1920, 1080  # По умолчанию

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_info()

# Основные настройки
APP_NAME = "Двухфакторная аутентификация"
VERSION = "1.1.0"

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATABASE_PATH = os.path.join(DATA_DIR, "users.db")

# Создание директорий если их нет
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Настройки машинного обучения
MIN_TRAINING_SAMPLES = 50
KNN_NEIGHBORS = 3
THRESHOLD_ACCURACY = 0.75

# Адаптивные настройки GUI под разрешение экрана
if SCREEN_WIDTH >= 1920 and SCREEN_HEIGHT >= 1080:
    # Для Full HD и выше
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 900
    TRAINING_WINDOW_WIDTH = 900
    TRAINING_WINDOW_HEIGHT = 1000
    STATS_WINDOW_WIDTH = 1400
    STATS_WINDOW_HEIGHT = 900
    FONT_SIZE = 11
elif SCREEN_WIDTH >= 1366:
    # Для HD экранов
    WINDOW_WIDTH = 700
    WINDOW_HEIGHT = 800
    TRAINING_WINDOW_WIDTH = 800
    TRAINING_WINDOW_HEIGHT = 900
    STATS_WINDOW_WIDTH = 1200
    STATS_WINDOW_HEIGHT = 800
    FONT_SIZE = 10
else:
    # Для маленьких экранов
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 700
    TRAINING_WINDOW_WIDTH = 700
    TRAINING_WINDOW_HEIGHT = 800
    STATS_WINDOW_WIDTH = 1000
    STATS_WINDOW_HEIGHT = 700
    FONT_SIZE = 9

FONT_FAMILY = "Arial"

# Настройки безопасности
SALT_LENGTH = 32

# Панграмма для обучения и аутентификации
PANGRAM = "The quick brown fox jumps over the lazy dog"

# Настройки для анализа и отладки
DEBUG_MODE = True
ENABLE_CSV_EXPORT = True
ENABLE_PERFORMANCE_TRACKING = True

# Пути для дополнительных данных
TEMP_DIR = os.path.join(DATA_DIR, "temp")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
CSV_EXPORTS_DIR = os.path.join(DATA_DIR, "csv_exports")

# Создание дополнительных директорий
os.makedirs(TEMP_DIR, exist_ok=True) 
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CSV_EXPORTS_DIR, exist_ok=True)

# Настройки аутентификации
DEFAULT_THRESHOLD = 0.75

# Эталонные показатели для сравнения
BENCHMARK_METRICS = {
    'commercial': {'far': 1.0, 'frr': 5.0, 'eer': 3.0},
    'research': {'far': 5.0, 'frr': 15.0, 'eer': 10.0},
    'acceptable': {'far': 10.0, 'frr': 25.0, 'eer': 15.0}
}

# Параметры модели по умолчанию
DEFAULT_KNN_PARAMS = {
    'n_neighbors': 3,
    'weights': 'distance',
    'metric': 'euclidean',
    'algorithm': 'ball_tree'
}

# Информация о конфигурации
if DEBUG_MODE:
    print(f"📋 Конфигурация {APP_NAME} v{VERSION}")
    print(f"🖥️  Экран: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"📐 Размеры окон: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    print(f"📁 Данные: {DATA_DIR}")