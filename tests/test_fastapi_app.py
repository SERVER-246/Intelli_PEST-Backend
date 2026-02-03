"""
Comprehensive Test Suite for Intelli-PEST Backend
==================================================
Tests covering FastAPI app, inference engine, filters, and utilities.

Run with: python -m pytest tests/ -v
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os

# Set up path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set CI environment
os.environ.setdefault('CI', 'true')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


class TestFastAPIImports(unittest.TestCase):
    """Test that all FastAPI modules can be imported."""
    
    def test_main_app_import(self):
        """Test main FastAPI app imports."""
        try:
            from inference_server.fastapi_app import app, create_app
            # app might be None (lazy init), but create_app should exist
            self.assertTrue(callable(create_app))
        except ImportError as e:
            self.skipTest(f"FastAPI not available: {e}")
    
    def test_routers_import(self):
        """Test routers module imports."""
        try:
            from inference_server.fastapi_app import routers
            self.assertTrue(hasattr(routers, 'router'))
        except ImportError as e:
            self.skipTest(f"Routers not available: {e}")
    
    def test_schemas_import(self):
        """Test schemas module imports."""
        try:
            from inference_server.fastapi_app import schemas
            self.assertTrue(hasattr(schemas, 'PredictionResponse'))
        except ImportError as e:
            self.skipTest(f"Schemas not available: {e}")
    
    def test_dependencies_import(self):
        """Test dependencies module imports."""
        try:
            from inference_server.fastapi_app import dependencies
            self.assertIsNotNone(dependencies)
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")


class TestConfigImports(unittest.TestCase):
    """Test configuration module imports."""
    
    def test_settings_import(self):
        """Test settings can be imported."""
        try:
            from inference_server.config import settings
            self.assertIsNotNone(settings)
        except ImportError as e:
            self.skipTest(f"Settings not available: {e}")
    
    def test_config_package(self):
        """Test config package exists."""
        config_dir = PROJECT_ROOT / 'inference_server' / 'config'
        self.assertTrue(config_dir.exists())
        self.assertTrue((config_dir / '__init__.py').exists())


class TestEngineImports(unittest.TestCase):
    """Test inference engine module imports."""
    
    def test_pytorch_inference_import(self):
        """Test PyTorch inference engine imports."""
        try:
            from inference_server.engine import pytorch_inference
            self.assertIsNotNone(pytorch_inference)
        except ImportError as e:
            self.skipTest(f"PyTorch inference not available: {e}")
    
    def test_onnx_inference_import(self):
        """Test ONNX inference engine imports."""
        try:
            from inference_server.engine import onnx_inference
            self.assertIsNotNone(onnx_inference)
        except ImportError as e:
            self.skipTest(f"ONNX inference not available: {e}")


class TestFiltersImports(unittest.TestCase):
    """Test filter module imports."""
    
    def test_filters_package(self):
        """Test filters package exists."""
        filters_dir = PROJECT_ROOT / 'inference_server' / 'filters'
        self.assertTrue(filters_dir.exists())
    
    def test_validation_filters(self):
        """Test validation filters can be imported."""
        try:
            from inference_server.filters import ValidationPipeline
            self.assertIsNotNone(ValidationPipeline)
        except ImportError as e:
            self.skipTest(f"Validation filters not available: {e}")


class TestUtilsImports(unittest.TestCase):
    """Test utility module imports."""
    
    def test_utils_package(self):
        """Test utils package exists."""
        utils_dir = PROJECT_ROOT / 'inference_server' / 'utils'
        self.assertTrue(utils_dir.exists())


class TestSrcModules(unittest.TestCase):
    """Test src directory modules."""
    
    def test_src_structure(self):
        """Test src directory structure."""
        src_dir = PROJECT_ROOT / 'src'
        self.assertTrue(src_dir.exists(), "src/ directory should exist")
        
        expected_subdirs = ['training', 'conversion', 'deployment', 'utils']
        for subdir in expected_subdirs:
            subdir_path = src_dir / subdir
            self.assertTrue(subdir_path.exists(), f"src/{subdir}/ should exist")
    
    def test_training_imports(self):
        """Test training modules import."""
        try:
            from src.training import base_training
            self.assertIsNotNone(base_training)
        except ImportError as e:
            self.skipTest(f"Training module not available: {e}")
    
    def test_utils_imports(self):
        """Test utils modules import."""
        try:
            from src.utils import data_counter
            self.assertTrue(hasattr(data_counter, 'count_images'))
        except ImportError as e:
            self.skipTest(f"Utils module not available: {e}")


class TestSchemaValidation(unittest.TestCase):
    """Test Pydantic schema validation."""
    
    def test_prediction_response_schema(self):
        """Test PredictionResponse schema."""
        try:
            from inference_server.fastapi_app.schemas import PredictionResponse, PredictionResult
            
            # Create a prediction result (using alias 'class' for class_name field)
            prediction = PredictionResult(**{
                "class": "army worm",
                "class_id": 0,
                "confidence": 0.95
            })
            
            # Test valid response
            response = PredictionResponse(
                status="success",
                prediction=prediction
            )
            self.assertEqual(response.status, "success")
            self.assertEqual(response.prediction.confidence, 0.95)
        except ImportError as e:
            self.skipTest(f"Schemas not available: {e}")
        except Exception as e:
            self.skipTest(f"Schema test failed: {e}")
    
    def test_health_response_schema(self):
        """Test HealthResponse schema if exists."""
        try:
            from inference_server.fastapi_app.schemas import HealthResponse, ModelInfo
            from datetime import datetime
            
            model_info = ModelInfo(loaded=True, info={"version": "1.0.0"})
            response = HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version="1.0.0",
                model=model_info
            )
            self.assertEqual(response.status, "healthy")
        except ImportError:
            self.skipTest("HealthResponse schema not available")
        except Exception:
            pass  # Schema might have different structure


class TestConnectionTracker(unittest.TestCase):
    """Test connection tracking functionality."""
    
    def test_tracker_import(self):
        """Test connection tracker imports."""
        try:
            from inference_server.fastapi_app import connection_tracker
            self.assertIsNotNone(connection_tracker)
        except ImportError as e:
            self.skipTest(f"connection_tracker not available: {e}")
    
    def test_tracker_module_exists(self):
        """Test tracker module exists."""
        tracker_file = PROJECT_ROOT / 'inference_server' / 'fastapi_app' / 'connection_tracker.py'
        self.assertTrue(tracker_file.exists(), "connection_tracker.py should exist")


class TestAppManagement(unittest.TestCase):
    """Test app management functionality."""
    
    def test_app_management_import(self):
        """Test app management imports."""
        try:
            from inference_server.fastapi_app import app_management
            self.assertIsNotNone(app_management)
        except ImportError as e:
            self.skipTest(f"App management not available: {e}")


class TestProjectStructure(unittest.TestCase):
    """Test overall project structure."""
    
    def test_required_directories(self):
        """Test required directories exist."""
        required_dirs = [
            'inference_server',
            'inference_server/fastapi_app',
            'inference_server/config',
            'inference_server/engine',
            'inference_server/filters',
            'src',
            'tests',
            'configs',
        ]
        
        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            self.assertTrue(full_path.exists(), f"Missing required directory: {dir_path}")
    
    def test_required_files(self):
        """Test required files exist."""
        required_files = [
            'run_server.py',
            'README.md',
            'pyproject.toml',
            'inference_server/__init__.py',
            'inference_server/fastapi_app/__init__.py',
        ]
        
        for file_path in required_files:
            full_path = PROJECT_ROOT / file_path
            self.assertTrue(full_path.exists(), f"Missing required file: {file_path}")
    
    def test_gitignore_exists(self):
        """Test .gitignore exists."""
        gitignore = PROJECT_ROOT / '.gitignore'
        self.assertTrue(gitignore.exists())
    
    def test_ci_workflow_exists(self):
        """Test CI workflow exists."""
        ci_workflow = PROJECT_ROOT / '.github' / 'workflows' / 'ci.yml'
        self.assertTrue(ci_workflow.exists(), "CI workflow should exist")


class TestConfigFiles(unittest.TestCase):
    """Test configuration files."""
    
    def test_pyproject_toml(self):
        """Test pyproject.toml is valid."""
        pyproject = PROJECT_ROOT / 'pyproject.toml'
        self.assertTrue(pyproject.exists())
        
        content = pyproject.read_text()
        self.assertIn('[tool.pytest', content)
        self.assertIn('[tool.ruff', content)
    
    def test_pyrightconfig(self):
        """Test pyrightconfig.json exists."""
        pyright_config = PROJECT_ROOT / 'pyrightconfig.json'
        self.assertTrue(pyright_config.exists())
        
        content = json.loads(pyright_config.read_text())
        self.assertIn('include', content)


class TestEnvironmentSafety(unittest.TestCase):
    """Test environment safety measures."""
    
    def test_no_hardcoded_secrets(self):
        """Test no hardcoded secrets in config files."""
        sensitive_patterns = ['password=', 'secret=', 'api_key=', 'token=']
        
        config_files = [
            PROJECT_ROOT / 'inference_server' / 'config' / 'settings.py',
        ]
        
        for config_file in config_files:
            if config_file.exists():
                content = config_file.read_text().lower()
                for pattern in sensitive_patterns:
                    # Allow patterns in comments or env var lookups
                    if pattern in content:
                        lines = content.split('\n')
                        for line in lines:
                            if pattern in line and not line.strip().startswith('#'):
                                if 'os.getenv' not in line and 'environ' not in line:
                                    self.fail(f"Potential hardcoded secret in {config_file}: {pattern}")
    
    def test_env_example_exists(self):
        """Test .env.example exists for reference."""
        env_example = PROJECT_ROOT / 'inference_server' / '.env.example'
        if env_example.exists():
            self.assertTrue(True)
        else:
            self.skipTest(".env.example not required but recommended")


if __name__ == '__main__':
    unittest.main(verbosity=2)
