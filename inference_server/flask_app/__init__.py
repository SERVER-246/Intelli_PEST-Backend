"""Flask Application."""
from .app import app
from .app import create_app as create_flask_app

__all__ = ["create_flask_app", "app"]
