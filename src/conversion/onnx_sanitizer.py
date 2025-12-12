"""
ONNX Model Sanitizer
====================
Fixes ONNX models with invalid TensorFlow scope names by sanitizing node names.

TensorFlow scope name requirements:
- Must match pattern: ^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$
- Cannot start with '/' or other special characters
- Must be valid identifiers

This utility cleans ONNX graphs to make them compatible with onnx2keras and TensorFlow.
"""

import re
import logging
from pathlib import Path
import onnx
from onnx import helper, checker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ONNXSanitizer')


class ONNXSanitizer:
    """Sanitizes ONNX models to be TensorFlow-compatible."""
    
    def __init__(self):
        self.name_mapping = {}
        self.counter = 0
    
    def sanitize_name(self, name: str) -> str:
        """
        Sanitize a node name to be TensorFlow-compatible.
        
        Rules:
        1. Remove leading slashes
        2. Replace remaining slashes with underscores
        3. Replace other invalid chars with underscores
        4. Ensure first character is alphanumeric
        5. Add prefix if name starts with number
        """
        if not name:
            return f"node_{self.counter}"
        
        # Check if already sanitized
        if name in self.name_mapping:
            return self.name_mapping[name]
        
        original = name
        
        # Remove leading slashes
        name = name.lstrip('/')
        
        # Replace slashes with underscores
        name = name.replace('/', '_')
        name = name.replace('\\', '_')
        
        # Replace other problematic characters
        name = re.sub(r'[^A-Za-z0-9_.\-]', '_', name)
        
        # Ensure it starts with letter or digit (not underscore)
        if name and not name[0].isalnum():
            name = 'node_' + name
        
        # Handle empty name after sanitization
        if not name:
            name = f"node_{self.counter}"
            self.counter += 1
        
        # Avoid duplicates
        base_name = name
        suffix = 0
        while name in self.name_mapping.values():
            suffix += 1
            name = f"{base_name}_{suffix}"
        
        self.name_mapping[original] = name
        return name
    
    def sanitize_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Sanitize all node names in an ONNX model.
        
        Args:
            model: Input ONNX model
            
        Returns:
            Sanitized ONNX model
        """
        logger.info("Sanitizing ONNX model...")
        
        # Create a copy of the model
        import copy
        sanitized_model = copy.deepcopy(model)
        
        graph = sanitized_model.graph
        
        # Sanitize node names
        for node in graph.node:
            if node.name:
                old_name = node.name
                node.name = self.sanitize_name(old_name)
                if old_name != node.name:
                    logger.debug(f"  Renamed node: {old_name} -> {node.name}")
        
        # Sanitize input names
        for inp in graph.input:
            if inp.name:
                old_name = inp.name
                new_name = self.sanitize_name(old_name)
                if old_name != new_name:
                    inp.name = new_name
                    logger.debug(f"  Renamed input: {old_name} -> {new_name}")
        
        # Sanitize output names
        for out in graph.output:
            if out.name:
                old_name = out.name
                new_name = self.sanitize_name(old_name)
                if old_name != new_name:
                    out.name = new_name
                    logger.debug(f"  Renamed output: {old_name} -> {new_name}")
        
        # Update all node inputs/outputs references
        for node in graph.node:
            # Update input references
            for i, inp in enumerate(node.input):
                if inp in self.name_mapping:
                    node.input[i] = self.name_mapping[inp]
            
            # Update output references
            for i, out in enumerate(node.output):
                if out in self.name_mapping:
                    old_out = out
                    node.output[i] = self.name_mapping[out]
                elif out:  # Sanitize if not yet in mapping
                    old_out = out
                    node.output[i] = self.sanitize_name(out)
                    if old_out != node.output[i]:
                        logger.debug(f"  Renamed output: {old_out} -> {node.output[i]}")
        
        # Sanitize initializer names
        for init in graph.initializer:
            if init.name in self.name_mapping:
                init.name = self.name_mapping[init.name]
        
        # Sanitize value_info names
        for val_info in graph.value_info:
            if val_info.name in self.name_mapping:
                val_info.name = self.name_mapping[val_info.name]
        
        logger.info(f"Sanitized {len(self.name_mapping)} names")
        
        # Fix Clip operators for onnx2keras compatibility
        sanitized_model = self._fix_clip_operators(sanitized_model)
        
        try:
            checker.check_model(sanitized_model)
            logger.info("✓ Sanitized model passes ONNX checker")
        except Exception as e:
            logger.warning(f"Sanitized model checker warning: {e}")
        
        return sanitized_model
    
    def _fix_clip_operators(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Fix Clip operators to be compatible with onnx2keras.
        
        In ONNX opset >= 11, Clip uses inputs for min/max.
        onnx2keras expects attributes. This converts Clip nodes to use attributes.
        """
        from onnx import numpy_helper
        import numpy as np
        
        graph = model.graph
        initializers = {init.name: init for init in graph.initializer}
        
        fixed_count = 0
        for node in graph.node:
            if node.op_type == 'Clip':
                min_val = 0.0  # Default for ReLU6
                max_val = 6.0  # Default for ReLU6
                
                # Try to get min/max from inputs
                if len(node.input) >= 2 and node.input[1] and node.input[1] in initializers:
                    min_tensor = initializers[node.input[1]]
                    min_val = float(numpy_helper.to_array(min_tensor).item())
                
                if len(node.input) >= 3 and node.input[2] and node.input[2] in initializers:
                    max_tensor = initializers[node.input[2]]
                    max_val = float(numpy_helper.to_array(max_tensor).item())
                
                # Clear inputs except the first one
                del node.input[1:]
                
                # Add attributes
                node.attribute.extend([
                    helper.make_attribute('min', min_val),
                    helper.make_attribute('max', max_val)
                ])
                
                fixed_count += 1
        
        if fixed_count > 0:
            logger.info(f"Fixed {fixed_count} Clip operators for onnx2keras compatibility")
        
        return model
    
    def sanitize_file(self, input_path: Path, output_path: Path = None) -> Path:
        """
        Sanitize an ONNX file and save it.
        
        Args:
            input_path: Path to input ONNX file
            output_path: Path to save sanitized model (if None, overwrites input)
            
        Returns:
            Path to sanitized model
        """
        logger.info(f"Loading ONNX model from: {input_path}")
        model = onnx.load(str(input_path))
        
        sanitized_model = self.sanitize_model(model)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_sanitized.onnx"
        
        logger.info(f"Saving sanitized model to: {output_path}")
        onnx.save(sanitized_model, str(output_path))
        
        # Verify saved model
        try:
            test_model = onnx.load(str(output_path))
            checker.check_model(test_model)
            logger.info("✓ Saved model verified successfully")
        except Exception as e:
            logger.error(f"✗ Saved model verification failed: {e}")
            raise
        
        return output_path


def sanitize_model_file(onnx_path: Path) -> Path:
    """
    Convenience function to sanitize a single ONNX file.
    
    Args:
        onnx_path: Path to ONNX file
        
    Returns:
        Path to sanitized ONNX file
    """
    sanitizer = ONNXSanitizer()
    sanitized_path = sanitizer.sanitize_file(onnx_path)
    return sanitized_path


def main():
    """Test sanitizer on blocked models."""
    import sys
    
    onnx_dir = Path(r"D:\Base-dir\onnx_models")
    
    # Models that need sanitization
    blocked_models = [
        'mobilenet_v2',
        'darknet53',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble'
    ]
    
    logger.info("="*70)
    logger.info("ONNX Model Sanitizer")
    logger.info("="*70)
    logger.info(f"Processing {len(blocked_models)} models\n")
    
    for model_name in blocked_models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Sanitizing: {model_name}")
        logger.info(f"{'='*70}")
        
        onnx_path = onnx_dir / model_name / f"{model_name}.onnx"
        
        if not onnx_path.exists():
            logger.warning(f"ONNX file not found: {onnx_path}")
            continue
        
        try:
            sanitizer = ONNXSanitizer()
            sanitized_path = sanitizer.sanitize_file(onnx_path)
            
            logger.info(f"✓ SUCCESS: {model_name}")
            logger.info(f"  Original:  {onnx_path}")
            logger.info(f"  Sanitized: {sanitized_path}")
            logger.info(f"  Renamed:   {len(sanitizer.name_mapping)} names")
            
        except Exception as e:
            logger.error(f"✗ FAILED: {model_name}")
            logger.exception(e)
    
    logger.info(f"\n{'='*70}")
    logger.info("Sanitization complete")
    logger.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()
