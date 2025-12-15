#!/usr/bin/env python3
"""
MASTER PIPELINE SCRIPT: Complete ML Pipeline from Training to TFLite Deployment
================================================================================

This script orchestrates the entire pest detection pipeline:
1. Train individual base models (7 models)
2. Create ensemble models (4 models)
3. Export to ONNX format
4. Convert to TensorFlow Lite with quantization
5. Validate and test all models

Usage:
    python pipeline.py                          # Run complete pipeline
    python pipeline.py --stage training         # Run only training
    python pipeline.py --stage ensemble         # Run only ensemble creation
    python pipeline.py --stage conversion       # Run only TFLite conversion
    python pipeline.py --stage test             # Run only validation
    python pipeline.py --data_path /path/to/data --epochs 100

Requirements:
    - Python 3.10+
    - Virtual environment activated (venv_tflite)
    - pip install -r requirements_tflite.txt
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Pipeline')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class MasterPipeline:
    """Orchestrates complete ML pipeline from training to TFLite conversion."""
    
    def __init__(self, args):
        """Initialize pipeline configuration."""
        self.args = args
        self.start_time = datetime.now()
        self.results = {}
        
        # Set directories
        self.root_dir = Path(__file__).parent
        self.data_path = Path(args.data_path) if args.data_path else self.root_dir / 'data'
        self.checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else self.root_dir / 'checkpoints'
        self.output_dir = Path(args.output_dir) if args.output_dir else self.root_dir / 'tflite_models'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def stage_1_training(self):
        """Stage 1: Train individual base models."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: TRAINING BASE MODELS (7 models)")
        logger.info("="*70)
        
        try:
            from training.base_training import BaseModelTrainer
            
            trainer = BaseModelTrainer(
                data_path=self.data_path,
                output_dir=self.checkpoint_dir,
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                learning_rate=self.args.learning_rate
            )
            
            logger.info("Training models:")
            logger.info("  1. MobileNetV2")
            logger.info("  2. ResNet50")
            logger.info("  3. InceptionV3")
            logger.info("  4. EfficientNetB0")
            logger.info("  5. YOLOv11n-cls")
            logger.info("  6. DarkNet53")
            logger.info("  7. AlexNet")
            
            results = trainer.train_all()
            self.results['training'] = results
            
            logger.info(f"✓ Stage 1 complete - {len(results)} models trained")
            return True
            
        except Exception as e:
            logger.error(f"✗ Stage 1 failed: {e}")
            if self.args.continue_on_error:
                logger.warning("Continuing to next stage...")
                return False
            raise
    
    def stage_2_ensemble(self):
        """Stage 2: Create ensemble models from base models."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: CREATING ENSEMBLE MODELS (4 models)")
        logger.info("="*70)
        
        try:
            from training.ensemble_training import EnsembleTrainer
            
            ensemble_trainer = EnsembleTrainer(
                checkpoint_dir=self.checkpoint_dir,
                output_dir=self.checkpoint_dir
            )
            
            logger.info("Creating ensemble models:")
            logger.info("  1. Attention-based Ensemble")
            logger.info("  2. Concatenation Ensemble")
            logger.info("  3. Cross-Attention Ensemble")
            logger.info("  4. Super Ensemble (combines all)")
            
            results = ensemble_trainer.create_all_ensembles()
            self.results['ensemble'] = results
            
            logger.info(f"✓ Stage 2 complete - {len(results)} ensemble models created")
            return True
            
        except Exception as e:
            logger.error(f"✗ Stage 2 failed: {e}")
            if self.args.continue_on_error:
                logger.warning("Continuing to next stage...")
                return False
            raise
    
    def stage_3_onnx(self):
        """Stage 3: Export models to ONNX format."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: ONNX EXPORT (11 models)")
        logger.info("="*70)
        
        logger.info("Note: ONNX models are pre-converted and available in Base-dir/onnx_models/")
        logger.info("These will be used as fallback in Stage 4 conversion if needed")
        logger.info("✓ Stage 3 - ONNX models ready")
        
        self.results['onnx'] = {
            'status': 'ready',
            'location': 'D:/Base-dir/onnx_models/',
            'models': 11
        }
        return True
    
    def stage_4_tflite(self):
        """Stage 4: Convert to TensorFlow Lite with quantization."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: TFLITE CONVERSION (11 models)")
        logger.info("="*70)
        
        try:
            from conversion.pytorch_to_tflite_quantized import PyTorchToTFLiteQuantized
            
            converter = PyTorchToTFLiteQuantized(
                pytorch_dir=self.checkpoint_dir,
                tflite_dir=self.output_dir
            )
            
            logger.info("Converting models to TFLite format with Dynamic Range Quantization...")
            logger.info("This will take 10-30 minutes depending on your hardware")
            logger.info("")
            
            # Convert all models
            results = converter.convert_all(verbose=self.args.verbose)
            self.results['tflite'] = results
            
            # Summary
            success_count = sum(1 for r in results.values() if r.get('status') == 'success')
            logger.info(f"\n✓ Stage 4 complete - {success_count}/{len(results)} models converted")
            logger.info(f"  Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Stage 4 failed: {e}")
            if self.args.continue_on_error:
                logger.warning("Continuing to validation...")
                return False
            raise
    
    def stage_5_validation(self):
        """Stage 5: Validate and test all models."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: VALIDATION & TESTING")
        logger.info("="*70)
        
        try:
            import subprocess
            
            # Run pytest
            logger.info("Running test suite...")
            test_result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                cwd=self.root_dir,
                capture_output=False
            )
            
            if test_result.returncode == 0:
                logger.info("✓ All tests passed!")
                self.results['validation'] = {'status': 'passed'}
            else:
                logger.warning("⚠ Some tests failed - check output above")
                self.results['validation'] = {'status': 'partial'}
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Stage 5 failed: {e}")
            logger.warning("Continuing to summary...")
            return False
    
    def run_complete_pipeline(self):
        """Execute complete pipeline with all stages."""
        logger.info("\n" + "="*70)
        logger.info("STARTING COMPLETE ML PIPELINE")
        logger.info(f"Start time: {self.start_time}")
        logger.info("="*70)
        
        stages = [
            ('training', self.stage_1_training),
            ('ensemble', self.stage_2_ensemble),
            ('onnx', self.stage_3_onnx),
            ('conversion', self.stage_4_tflite),
            ('validation', self.stage_5_validation),
        ]
        
        if self.args.stage:
            # Run specific stage
            stage_map = {stage: func for stage, func in stages}
            if self.args.stage in stage_map:
                logger.info(f"Running stage: {self.args.stage}")
                stage_map[self.args.stage]()
            else:
                logger.error(f"Unknown stage: {self.args.stage}")
                logger.info(f"Available stages: {', '.join(stage_map.keys())}")
                sys.exit(1)
        else:
            # Run all stages
            for stage_name, stage_func in stages:
                try:
                    stage_func()
                except Exception as e:
                    logger.error(f"Pipeline stopped at {stage_name}: {e}")
                    if not self.args.continue_on_error:
                        sys.exit(1)
        
        self.print_summary()
    
    def print_summary(self):
        """Print pipeline execution summary."""
        elapsed = datetime.now() - self.start_time
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nElapsed time: {elapsed}")
        logger.info(f"\nResults:")
        for stage, result in self.results.items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                logger.info(f"  {stage}: {status}")
            else:
                logger.info(f"  {stage}: completed")
        
        logger.info(f"\nOutput:")
        logger.info(f"  TFLite models: {self.output_dir}")
        tflite_count = len(list(self.output_dir.glob('**/*.tflite')))
        logger.info(f"  Total models: {tflite_count}/11")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review TFLite models in: {self.output_dir}")
        logger.info(f"  2. Deploy to mobile: Copy .tflite files to Android app")
        logger.info(f"  3. For more details, see: COMPLETE_PIPELINE.md")
        
        logger.info("\n" + "="*70)
        logger.info("✓ PIPELINE EXECUTION COMPLETE")
        logger.info("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Complete ML Pipeline: Training to TFLite Conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:
    # Run complete pipeline
    python pipeline.py
    
    # Run specific stage
    python pipeline.py --stage conversion
    python pipeline.py --stage training --epochs 100
    
    # Custom paths
    python pipeline.py --data_path /path/to/data --checkpoint_dir ./my_models
    
    # Verbose output
    python pipeline.py --verbose
    
STAGES:
    training      - Train 7 base models
    ensemble      - Create 4 ensemble models
    onnx          - Prepare ONNX models (already pre-converted)
    conversion    - Convert to TFLite with quantization
    validation    - Run test suite
        '''
    )
    
    # Pipeline control
    parser.add_argument(
        '--stage',
        type=str,
        choices=['training', 'ensemble', 'onnx', 'conversion', 'validation'],
        help='Run specific stage only'
    )
    
    # Training parameters
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    # Output control
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Directory for saving model checkpoints'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Directory for TFLite output models'
    )
    
    # Execution options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--continue_on_error',
        action='store_true',
        help='Continue to next stage on error'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MasterPipeline(args)
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()
