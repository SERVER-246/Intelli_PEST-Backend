"""Test 2G/slow connection improvements"""
import requests
import json
from pathlib import Path
import time

base = 'http://localhost:8000'
API_KEY = 'ip_test_key_intelli_pest_2025'

print('='*60)
print('TESTING 2G/SLOW CONNECTION IMPROVEMENTS')
print('='*60)

# Find a test image
test_img = None
for path in [Path('D:/Test-images/Top borer'), Path('D:/Test-images/Stalk borer')]:
    if path.exists():
        imgs = list(path.glob('*.jpg'))[:1]
        if imgs:
            test_img = imgs[0]
            break

if not test_img:
    print('No test image found!')
    exit(1)

print(f'\nUsing test image: {test_img.name}')

# Test 1: Normal prediction (full response)
print('\n1. FULL RESPONSE (normal mode)')
print('-'*40)
with open(test_img, 'rb') as f:
    headers = {
        'X-API-Key': API_KEY,
        'Accept-Encoding': 'gzip, deflate'
    }
    start = time.time()
    r = requests.post(f'{base}/api/v1/predict?include_probabilities=true', 
                     files={'image': f}, headers=headers, timeout=60)
    elapsed = time.time() - start

content_encoding = r.headers.get('Content-Encoding', 'none')
response_size = len(r.content)
print(f'   Status: {r.status_code}')
print(f'   Content-Encoding: {content_encoding}')
print(f'   Response size: {response_size} bytes')
print(f'   Time: {elapsed*1000:.0f}ms')

if r.status_code == 200:
    data = r.json()
    pred = data.get('prediction', {})
    print(f'   Prediction: {pred.get("class")}')
    print(f'   Has Phase3: {"phase3" in data and data["phase3"] is not None}')
    print(f'   Has inference: {"inference" in data and data["inference"] is not None}')

# Test 2: Lite mode prediction
print('\n2. LITE MODE (minimal response for 2G)')
print('-'*40)
with open(test_img, 'rb') as f:
    headers = {
        'X-API-Key': API_KEY,
        'Accept-Encoding': 'gzip, deflate'
    }
    start = time.time()
    r_lite = requests.post(f'{base}/api/v1/predict?lite=true', 
                          files={'image': f}, headers=headers, timeout=60)
    elapsed_lite = time.time() - start

content_encoding_lite = r_lite.headers.get('Content-Encoding', 'none')
response_size_lite = len(r_lite.content)
print(f'   Status: {r_lite.status_code}')
print(f'   Content-Encoding: {content_encoding_lite}')
print(f'   Response size: {response_size_lite} bytes')
print(f'   Time: {elapsed_lite*1000:.0f}ms')

if r_lite.status_code == 200:
    data_lite = r_lite.json()
    pred_lite = data_lite.get('prediction', {})
    print(f'   Prediction: {pred_lite.get("class")}')
    print(f'   Has Phase3: {"phase3" in data_lite and data_lite["phase3"] is not None}')
    print(f'   Has inference: {"inference" in data_lite and data_lite["inference"] is not None}')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'   Full response:  {response_size} bytes')
print(f'   Lite response:  {response_size_lite} bytes')
if response_size > 0:
    savings = (1 - response_size_lite/response_size) * 100
    print(f'   Size reduction: {savings:.1f}%')

print()
print('2G SIMULATION (50 Kbps = 6.25 KB/s):')
print(f'   Full response download: {response_size/6250:.2f}s')
print(f'   Lite response download: {response_size_lite/6250:.2f}s')

if content_encoding == 'gzip':
    print('\n✅ GZIP compression: WORKING')
else:
    print('\n⚠️  GZIP: Not applied (response below 500 byte threshold)')

if response_size_lite < response_size:
    print('✅ Lite mode: WORKING (smaller response)')
else:
    print('❌ Lite mode: Not reducing size')
