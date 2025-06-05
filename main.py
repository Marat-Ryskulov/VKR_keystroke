#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система двухфакторной аутентификации с динамикой нажатий клавиш
Главный файл запуска приложения
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь Python
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Импортируем главное окно
from gui.main_window import MainWindow

def main():
    """Точка входа в приложение"""
    try:
        # Создаем и запускаем главное окно
        app = MainWindow()
        app.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()