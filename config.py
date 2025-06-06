# config.py - Конфигурационный файл системы

import os

# Основные настройки
APP_NAME = "Двухфакторная аутентификация"
VERSION = "1.0.0"

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATABASE_PATH = os.path.join(DATA_DIR, "users.db")

# Создание директорий если их нет
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Настройки машинного обучения
MIN_TRAINING_SAMPLES = 50  # Минимум образцов для обучения
KNN_NEIGHBORS = 3          # Количество соседей для KNN
THRESHOLD_ACCURACY = 0.80  # Порог точности для аутентификации

# Настройки GUI
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 700
FONT_FAMILY = "Arial"
FONT_SIZE = 12

# Настройки безопасности
SALT_LENGTH = 32

# Панграмма для обучения и аутентификации
PANGRAM = "The quick brown fox jumps over the lazy dog"

# Параметры динамики нажатий
KEYSTROKE_FEATURES = [
    'dwell_time',      # Время удержания клавиши
    'flight_time',     # Время между клавишами
    'typing_speed',    # Общая скорость печати
    'pressure_time'    # Время нажатия (для анализа ритма)
]