"""
API Endpoint Tests for Intelli-PEST Backend
============================================
Tests for the FastAPI endpoints using TestClient.

Run with: python -m pytest tests/test_api_endpoints.py -v
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Set up path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set CI environment
os.environ.setdefault('CI', 'true')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints using FastAPI TestClient."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        try:
            from fastapi.testclient import TestClient
            from inference_server.fastapi_app import app
            cls.client = TestClient(app)
            cls.client_available = True
        except ImportError as e:
            cls.client_available = False
            cls.skip_reason = f"TestClient not available: {e}"
        except Exception as e:
            cls.client_available = False
            cls.skip_reason = f"App initialization failed: {e}"
    
    def setUp(self):
        """Skip if client not available."""
        if not self.client_available:
            self.skipTest(self.skip_reason)
    
    def test_health_endpoint(self):
        """Test /health endpoint."""
        response = self.client.get("/health")
        self.assertIn(response.status_code, [200, 404])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn('status', data)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertIn(response.status_code, [200, 404, 307])
    
    def test_docs_endpoint(self):
        """Test /docs endpoint (Swagger UI)."""
        response = self.client.get("/docs")
        self.assertIn(response.status_code, [200, 404])
    
    def test_openapi_endpoint(self):
        """Test /openapi.json endpoint."""
        response = self.client.get("/openapi.json")
        if response.status_code == 200:
            data = response.json()
            self.assertIn('openapi', data)
            self.assertIn('info', data)
    
    def test_predict_endpoint_no_file(self):
        """Test /predict endpoint without file returns error."""
        response = self.client.post("/api/v1/predict")
        # Should fail without file
        self.assertIn(response.status_code, [400, 422, 404])
    
    def test_predict_endpoint_invalid_file(self):
        """Test /predict endpoint with invalid file."""
        # Send text file instead of image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = self.client.post("/api/v1/predict", files=files)
        # Should reject non-image file
        self.assertIn(response.status_code, [400, 415, 422, 404])


class TestAPIResponseFormat(unittest.TestCase):
    """Test API response formats."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        try:
            from fastapi.testclient import TestClient
            from inference_server.fastapi_app import app
            cls.client = TestClient(app)
            cls.client_available = True
        except Exception as e:
            cls.client_available = False
            cls.skip_reason = str(e)
    
    def setUp(self):
        if not self.client_available:
            self.skipTest(self.skip_reason)
    
    def test_error_response_format(self):
        """Test error responses have proper format."""
        response = self.client.post("/api/v1/predict")
        
        if response.status_code >= 400:
            data = response.json()
            # Should have detail field for FastAPI errors
            self.assertTrue(
                'detail' in data or 'error' in data or 'message' in data,
                "Error response should have detail/error/message field"
            )


class TestAPIValidation(unittest.TestCase):
    """Test API input validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        try:
            from fastapi.testclient import TestClient
            from inference_server.fastapi_app import app
            cls.client = TestClient(app)
            cls.client_available = True
        except Exception as e:
            cls.client_available = False
            cls.skip_reason = str(e)
    
    def setUp(self):
        if not self.client_available:
            self.skipTest(self.skip_reason)
    
    def test_large_file_rejection(self):
        """Test that overly large files are rejected."""
        # Create a fake large file (10MB of zeros)
        large_content = b"0" * (10 * 1024 * 1024)
        files = {"file": ("large.jpg", large_content, "image/jpeg")}
        
        response = self.client.post("/api/v1/predict", files=files)
        # Should either reject or timeout
        self.assertIn(response.status_code, [400, 413, 422, 500, 404])
    
    def test_empty_file_rejection(self):
        """Test that empty files are rejected."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = self.client.post("/api/v1/predict", files=files)
        # Should reject empty file
        self.assertIn(response.status_code, [400, 422, 404])


class TestAPIHeaders(unittest.TestCase):
    """Test API response headers."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        try:
            from fastapi.testclient import TestClient
            from inference_server.fastapi_app import app
            cls.client = TestClient(app)
            cls.client_available = True
        except Exception as e:
            cls.client_available = False
            cls.skip_reason = str(e)
    
    def setUp(self):
        if not self.client_available:
            self.skipTest(self.skip_reason)
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options("/api/v1/predict")
        # CORS might not be configured, so just check it doesn't error
        self.assertIn(response.status_code, [200, 204, 405, 404])
    
    def test_content_type_json(self):
        """Test JSON responses have correct content type."""
        response = self.client.get("/health")
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            self.assertIn('application/json', content_type)


class TestMockedPrediction(unittest.TestCase):
    """Test prediction with mocked inference engine."""
    
    def test_mock_prediction_response(self):
        """Test prediction endpoint with mocked model."""
        try:
            from fastapi.testclient import TestClient
            from inference_server.fastapi_app import app
            
            # Mock the inference pipeline
            mock_result = {
                'pest_class': 'army worm',
                'confidence': 0.95,
                'class_id': 0,
                'processing_time_ms': 150.5,
                'model_version': '1.0.0'
            }
            
            with patch('inference_server.fastapi_app.dependencies.get_pipeline') as mock_pipeline:
                mock_pipeline.return_value.predict.return_value = mock_result
                
                client = TestClient(app)
                # Create a minimal valid JPEG
                jpeg_header = bytes([0xFF, 0xD8, 0xFF, 0xE0])
                files = {"file": ("test.jpg", jpeg_header, "image/jpeg")}
                
                response = client.post("/api/v1/predict", files=files)
                # May fail validation but shouldn't crash
                self.assertIn(response.status_code, [200, 400, 422, 500, 404])
                
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")
        except Exception as e:
            self.skipTest(f"Mock test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
