# gui/__init__.py
from .main_window import MainWindow
from .login_window import LoginWindow
from .register_window import RegisterWindow
from .training_window import TrainingWindow
from .model_stats_window import ModelStatsWindow

__all__ = ['MainWindow', 'LoginWindow', 'RegisterWindow', 'TrainingWindow', 
           'ModelStatsWindow']