"""Test slow connection support features"""
import requests
import json
import time
from pathlib import Path

base = 'http://localhost:8000'
API_KEY = 'ip_test_key_intelli_pest_2025'

print('='*60)
print('TESTING SLOW CONNECTION SUPPORT')
print('='*60)

# Test 1: Connection sample reporting
print('\n1. TESTING CONNECTION SAMPLE REPORTING')
print('-'*40)
payload = {
    'device_id': 'test_device_001',
    'user_id': 'test_user_001',
    'app_version': '1.0.0',
    'samples': [
        {
            'timestamp': int(time.time() * 1000),
            'network_type': '2g',
            'quality_level': 2,
            'download_speed_kbps': 45,
            'latitude': 26.9124,
            'longitude': 75.7873
        },
        {
            'timestamp': int(time.time() * 1000) - 60000,
            'network_type': '3g',
            'quality_level': 3,
            'download_speed_kbps': 350,
            'latitude': 26.9125,
            'longitude': 75.7874
        },
        {
            'timestamp': int(time.time() * 1000) - 120000,
            'network_type': '4g',
            'quality_level': 4,
            'download_speed_kbps': 5000,
            'latitude': 26.9126,
            'longitude': 75.7875
        }
    ]
}

r = requests.post(f'{base}/api/v1/connection/report',
                  json=payload,
                  headers={'X-API-Key': API_KEY},
                  timeout=30)
print(f'   Status: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    print(f'   Samples received: {data.get("samples_received", 0)}')
    print('   ✅ SUCCESS')
else:
    print(f'   ❌ Error: {r.text[:200]}')

# Test 2: Prediction with X-Connection-Info header
print('\n2. TESTING PREDICTION WITH CONNECTION INFO HEADER')
print('-'*40)
test_img = list(Path('D:/Test-images/Top borer').glob('*.jpg'))[0]
with open(test_img, 'rb') as f:
    headers = {
        'X-API-Key': API_KEY,
        'X-Connection-Info': 'type:2g;quality:2;speed:45',
        'Accept-Encoding': 'gzip, deflate'
    }
    r = requests.post(f'{base}/api/v1/predict', 
                     files={'image': f}, headers=headers, timeout=60)
print(f'   Status: {r.status_code}')
print(f'   GZIP: {r.headers.get("Content-Encoding", "none")}')
if r.status_code == 200:
    print('   ✅ SUCCESS')
else:
    print(f'   ❌ Error: {r.text[:200]}')

# Test 3: Lite mode
print('\n3. TESTING LITE MODE')
print('-'*40)
with open(test_img, 'rb') as f:
    r = requests.post(f'{base}/api/v1/predict?lite=true', 
                     files={'image': f}, 
                     headers={'X-API-Key': API_KEY},
                     timeout=60)
print(f'   Status: {r.status_code}')
print(f'   Response size: {len(r.content)} bytes')
if r.status_code == 200:
    data = r.json()
    print(f'   Has inference: {data.get("inference") is not None}')
    print(f'   Has phase3: {data.get("phase3") is not None}')
    print('   ✅ SUCCESS')
else:
    print(f'   ❌ Error: {r.text[:200]}')

# Test 4: Compare full vs lite response sizes
print('\n4. COMPARING FULL VS LITE RESPONSE SIZES')
print('-'*40)
with open(test_img, 'rb') as f:
    r_full = requests.post(f'{base}/api/v1/predict?include_probabilities=true', 
                          files={'image': f}, 
                          headers={'X-API-Key': API_KEY, 'Accept-Encoding': 'gzip'},
                          timeout=60)
with open(test_img, 'rb') as f:
    r_lite = requests.post(f'{base}/api/v1/predict?lite=true', 
                          files={'image': f}, 
                          headers={'X-API-Key': API_KEY, 'Accept-Encoding': 'gzip'},
                          timeout=60)
full_size = len(r_full.content)
lite_size = len(r_lite.content)
reduction = (1 - lite_size/full_size) * 100 if full_size > 0 else 0

print(f'   Full response: {full_size} bytes')
print(f'   Lite response: {lite_size} bytes')
print(f'   Size reduction: {reduction:.1f}%')

# 2G simulation
print(f'\n   2G Download Time (50 Kbps):')
print(f'   Full: {full_size/6250:.2f}s')
print(f'   Lite: {lite_size/6250:.2f}s')

print('\n' + '='*60)
print('ALL TESTS COMPLETE')
print('='*60)
