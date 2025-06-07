# gui/__init__.py - Исправленный импорт GUI модулей
from .main_window import MainWindow
from .login_window import LoginWindow
from .register_window import RegisterWindow
from .training_window import TrainingWindow
from .model_stats_window import ModelStatsWindow
from .enhanced_model_stats_window import EnhancedModelStatsWindow

__all__ = [
    'MainWindow', 
    'LoginWindow', 
    'RegisterWindow', 
    'TrainingWindow', 
    'ModelStatsWindow',
    'EnhancedModelStatsWindow'
]