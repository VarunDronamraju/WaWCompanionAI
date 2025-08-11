"""
Backend Package Initialization
Location: backend/__init__.py
Phase: 3 (Python package structure)
"""

__version__ = "1.0.0"
__author__ = "RAG Desktop Development Team"
__description__ = "RAG Desktop Application Backend"

# Import main components for easy access
from .main import app, create_app, get_app_info
from .config import get_settings, Settings
from .schemas import *
from .utils import *

__all__ = [
    "app",
    "create_app", 
    "get_app_info",
    "get_settings",
    "Settings"
]