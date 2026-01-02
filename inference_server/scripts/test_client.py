#!/usr/bin/env python3
"""
Test Client Script
==================
Test client for the inference server API.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path


def test_health(base_url: str, api_key: str = None) -> dict:
    """Test health endpoint."""
    import requests
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    response = requests.get(f"{base_url}/api/v1/health", headers=headers)
    return response.json()


def test_classes(base_url: str, api_key: str = None) -> dict:
    """Test classes endpoint."""
    import requests
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    response = requests.get(f"{base_url}/api/v1/classes", headers=headers)
    return response.json()


def test_predict_file(base_url: str, image_path: str, api_key: str = None) -> dict:
    """Test prediction with file upload."""
    import requests
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(
            f"{base_url}/api/v1/predict",
            files=files,
            headers=headers,
            params={"include_probabilities": "true"},
        )
    
    return response.json()


def test_predict_base64(base_url: str, image_path: str, api_key: str = None) -> dict:
    """Test prediction with base64 encoded image."""
    import requests
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        f"{base_url}/api/v1/predict/base64",
        json={"image_data": image_data, "include_probabilities": True},
        headers=headers,
    )
    
    return response.json()


def test_batch(base_url: str, image_paths: list, api_key: str = None) -> dict:
    """Test batch prediction."""
    import requests
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    files = []
    for path in image_paths:
        with open(path, "rb") as f:
            files.append(("images", (Path(path).name, f.read(), "image/jpeg")))
    
    response = requests.post(
        f"{base_url}/api/v1/predict/batch",
        files=files,
        headers=headers,
    )
    
    return response.json()


def run_all_tests(base_url: str, image_path: str = None, api_key: str = None):
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("INFERENCE SERVER API TEST")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"API Key: {'Provided' if api_key else 'Not provided'}")
    print("=" * 60)
    
    # Test health
    print("\n[1/4] Testing Health Endpoint...")
    try:
        result = test_health(base_url, api_key)
        status = result.get("status", "unknown")
        print(f"  Status: {status}")
        print(f"  Model Loaded: {result.get('model', {}).get('loaded', False)}")
        print("  ✓ Health check passed")
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return
    
    # Test classes
    print("\n[2/4] Testing Classes Endpoint...")
    try:
        result = test_classes(base_url, api_key)
        num_classes = result.get("num_classes", 0)
        print(f"  Number of classes: {num_classes}")
        print("  ✓ Classes endpoint passed")
    except Exception as e:
        print(f"  ✗ Classes endpoint failed: {e}")
    
    # Test prediction
    if image_path:
        print(f"\n[3/4] Testing Prediction (File Upload)...")
        print(f"  Image: {image_path}")
        try:
            start = time.time()
            result = test_predict_file(base_url, image_path, api_key)
            elapsed = (time.time() - start) * 1000
            
            if result.get("status") == "success":
                pred = result.get("prediction", {})
                print(f"  Predicted Class: {pred.get('class', 'N/A')}")
                print(f"  Confidence: {pred.get('confidence', 0):.2%}")
                print(f"  Response Time: {elapsed:.0f}ms")
                print("  ✓ Prediction passed")
            else:
                print(f"  ✗ Prediction failed: {result.get('error', {}).get('message', 'Unknown error')}")
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
        
        print(f"\n[4/4] Testing Prediction (Base64)...")
        try:
            start = time.time()
            result = test_predict_base64(base_url, image_path, api_key)
            elapsed = (time.time() - start) * 1000
            
            if result.get("status") == "success":
                pred = result.get("prediction", {})
                print(f"  Predicted Class: {pred.get('class', 'N/A')}")
                print(f"  Confidence: {pred.get('confidence', 0):.2%}")
                print(f"  Response Time: {elapsed:.0f}ms")
                print("  ✓ Base64 prediction passed")
            else:
                print(f"  ✗ Base64 prediction failed: {result.get('error', {}).get('message', 'Unknown error')}")
        except Exception as e:
            print(f"  ✗ Base64 prediction failed: {e}")
    else:
        print("\n[3/4] Skipping Prediction (No image provided)")
        print("[4/4] Skipping Base64 Prediction (No image provided)")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test Client for Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test all:      python test_client.py --url http://localhost:8000 --image test.jpg
  Test health:   python test_client.py --url http://localhost:8000 --test health
  Test predict:  python test_client.py --url http://localhost:8000 --test predict --image test.jpg
        """,
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["health", "classes", "predict", "batch", "all"],
        default="all",
        help="Test to run (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON responses",
    )
    
    args = parser.parse_args()
    
    # Check requests is installed
    try:
        import requests
    except ImportError:
        print("Error: requests library is required. Install with: pip install requests")
        sys.exit(1)
    
    base_url = args.url.rstrip("/")
    
    if args.test == "all":
        run_all_tests(base_url, args.image, args.api_key)
    
    elif args.test == "health":
        result = test_health(base_url, args.api_key)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Status: {result.get('status')}")
            print(f"Model Loaded: {result.get('model', {}).get('loaded')}")
    
    elif args.test == "classes":
        result = test_classes(base_url, args.api_key)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for cls in result.get("classes", []):
                print(f"  {cls['id']}: {cls['name']}")
    
    elif args.test == "predict":
        if not args.image:
            print("Error: --image is required for predict test")
            sys.exit(1)
        result = test_predict_file(base_url, args.image, args.api_key)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            pred = result.get("prediction", {})
            print(f"Class: {pred.get('class')}")
            print(f"Confidence: {pred.get('confidence', 0):.2%}")


if __name__ == "__main__":
    main()
