"""Phase 3 Integration Unit Tests"""
import sys
sys.path.insert(0, 'black_ops_training')

print('='*70)
print('  PHASE 3 INTEGRATION UNIT TESTS')
print('='*70)

passed = 0
failed = 0

# TEST 1: Model Path Consistency
print('\n[TEST 1] Model Path Consistency Across All Training Systems')
try:
    import re
    paths = {}
    with open('run_server.py', 'r') as f:
        m = re.search(r'"pytorch":\s*r?"([^"]+)"', f.read())
        paths['server'] = m.group(1) if m else None
    
    from ghost_config import BlackOpsConfig
    paths['ghost'] = BlackOpsConfig().deployed_model_path
    
    from inference_server.training.comprehensive_trainer import ComprehensiveTrainingConfig
    paths['comprehensive'] = ComprehensiveTrainingConfig().model_path
    
    norm = lambda p: p.replace('\\', '/').lower()
    unique = set(norm(v) for v in paths.values() if v)
    
    if len(unique) == 1:
        print(f'  PASS: All systems use same path')
        passed += 1
    else:
        print(f'  FAIL: Path mismatch: {paths}')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 2: Server loads correct model
print('\n[TEST 2] Server Model File Exists')
try:
    from pathlib import Path
    model_path = Path(r'D:\KnowledgeDistillation\student_model_rotation_robust.pt')
    if model_path.exists():
        print(f'  PASS: Model exists')
        passed += 1
    else:
        print(f'  FAIL: Model not found')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 3: Model has 12 classes with junk
print('\n[TEST 3] Model Has 12 Classes (Including Junk)')
try:
    import torch
    ckpt = torch.load(r'D:\KnowledgeDistillation\student_model_rotation_robust.pt', map_location='cpu')
    classes = ckpt.get('class_names', [])
    has_junk = 'junk' in classes
    has_12 = len(classes) == 12
    
    if has_junk and has_12:
        print(f'  PASS: 12 classes with junk')
        passed += 1
    else:
        print(f'  FAIL: Classes={len(classes)}, junk={has_junk}')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 4: Ghost Config Phase 3 Enabled
print('\n[TEST 4] Ghost Config Phase 3 Enabled')
try:
    from ghost_config import BlackOpsConfig
    cfg = BlackOpsConfig()
    if cfg.enable_phase3 and cfg.include_junk_class:
        print(f'  PASS: Phase3 and Junk enabled')
        passed += 1
    else:
        print(f'  FAIL: Phase3={cfg.enable_phase3}, Junk={cfg.include_junk_class}')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 5: Training Utils Has 12 Classes
print('\n[TEST 5] Training Utils STUDENT_NUM_CLASSES = 12')
try:
    from training_utils import STUDENT_NUM_CLASSES, CANONICAL_CLASSES_WITH_JUNK
    if STUDENT_NUM_CLASSES == 12 and len(CANONICAL_CLASSES_WITH_JUNK) == 12:
        print(f'  PASS: STUDENT_NUM_CLASSES=12')
        passed += 1
    else:
        print(f'  FAIL: STUDENT_NUM_CLASSES={STUDENT_NUM_CLASSES}')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 6: API Schemas Exist
print('\n[TEST 6] Phase 3 API Schemas Exist')
try:
    from inference_server.fastapi_app.schemas import (
        Phase3Response, Phase3AttentionInfo, Phase3RegionInfo,
        Phase3MultiLabelPrediction, PredictionResponse
    )
    has_phase3_field = 'phase3' in PredictionResponse.model_fields
    if has_phase3_field:
        print('  PASS: All Phase3 schemas present')
        passed += 1
    else:
        print('  FAIL: PredictionResponse missing phase3 field')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 7: Inference Engine Phase 3 Available
print('\n[TEST 7] Inference Engine Phase 3 Integration')
try:
    from inference_server.engine.inference import PHASE3_AVAILABLE
    if PHASE3_AVAILABLE:
        print('  PASS: PHASE3_AVAILABLE=True')
        passed += 1
    else:
        print('  FAIL: PHASE3_AVAILABLE=False')
        failed += 1
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# TEST 8: Live API /classes endpoint
print('\n[TEST 8] Live API /classes Endpoint')
try:
    import requests
    r = requests.get('http://localhost:8000/api/v1/classes', timeout=5)
    if r.status_code == 200:
        data = r.json()
        classes = data.get('classes', [])
        class_names = [c['name'] for c in classes] if isinstance(classes[0], dict) else classes
        if 'junk' in class_names and len(class_names) == 12:
            print(f'  PASS: API returns 12 classes with junk')
            passed += 1
        else:
            print(f'  FAIL: API classes={len(class_names)}, junk={"junk" in class_names}')
            failed += 1
    else:
        print(f'  SKIP: Server returned {r.status_code}')
except requests.exceptions.ConnectionError:
    print('  SKIP: Server not running')
except Exception as e:
    print(f'  FAIL: {e}')
    failed += 1

# SUMMARY
print('\n' + '='*70)
print('  TEST SUMMARY')
print('='*70)
print(f'  Passed: {passed}')
print(f'  Failed: {failed}')
if failed == 0:
    print('\n  [OK] ALL TESTS PASSED')
else:
    print(f'\n  [WARN] {failed} test(s) failed')
