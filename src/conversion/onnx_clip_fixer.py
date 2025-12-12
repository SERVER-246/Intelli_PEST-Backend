"""
Fix ONNX Clip operator for onnx2keras compatibility
====================================================
The 'Clip' operator in ONNX opset >= 11 uses inputs instead of attributes for min/max,
but onnx2keras expects attributes. This script converts Clip nodes to use attributes.
"""

import onnx
from onnx import numpy_helper
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ONNXClipFixer')


def fix_clip_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Fix Clip nodes to be compatible with onnx2keras.
    
    In ONNX opset >= 11, Clip uses inputs for min/max.
    onnx2keras expects attributes (opset < 11 style).
    This function converts the inputs to attributes.
    """
    graph = model.graph
    
    # Get initializers as a dict for easy lookup
    initializers = {init.name: init for init in graph.initializer}
    
    nodes_to_remove = []
    nodes_to_add = []
    
    for i, node in enumerate(graph.node):
        if node.op_type == 'Clip':
            logger.debug(f"Found Clip node: {node.name}")
            
            # Get min and max values from inputs
            min_val = None
            max_val = None
            
            # Clip node inputs: [input, min, max]
            if len(node.input) >= 2 and node.input[1]:
                min_input = node.input[1]
                if min_input in initializers:
                    min_tensor = initializers[min_input]
                    min_val = numpy_helper.to_array(min_tensor).item()
                    logger.debug(f"  Found min: {min_val}")
            
            if len(node.input) >= 3 and node.input[2]:
                max_input = node.input[2]
                if max_input in initializers:
                    max_tensor = initializers[max_input]
                    max_val = numpy_helper.to_array(max_tensor).item()
                    logger.debug(f"  Found max: {max_val}")
            
            # Create new Clip node with attributes
            new_node = onnx.helper.make_node(
                'Clip',
                inputs=[node.input[0]],  # Only keep the first input
                outputs=node.output,
                name=node.name
            )
            
            # Add min/max as attributes
            if min_val is not None:
                new_node.attribute.append(
                    onnx.helper.make_attribute('min', float(min_val))
                )
            if max_val is not None:
                new_node.attribute.append(
                    onnx.helper.make_attribute('max', float(max_val))
                )
            
            nodes_to_remove.append(i)
            nodes_to_add.append((i, new_node))
            
            logger.info(f"Fixed Clip node: {node.name} (min={min_val}, max={max_val})")
    
    # Replace nodes
    if nodes_to_remove:
        new_nodes = list(graph.node)
        for idx, new_node in reversed(nodes_to_add):
            new_nodes[idx] = new_node
        
        # Clear and rebuild
        del graph.node[:]
        graph.node.extend(new_nodes)
        
        logger.info(f"Fixed {len(nodes_to_remove)} Clip nodes")
    else:
        logger.info("No Clip nodes needed fixing")
    
    return model


def fix_onnx_for_onnx2keras(onnx_path: Path, output_path: Path = None) -> Path:
    """
    Fix an ONNX model to be compatible with onnx2keras.
    
    Args:
        onnx_path: Path to input ONNX file
        output_path: Path to save fixed model (if None, overwrites input)
        
    Returns:
        Path to fixed model
    """
    logger.info(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))
    
    logger.info(f"Model opset version: {model.opset_import[0].version}")
    
    # Fix Clip nodes
    model = fix_clip_nodes(model)
    
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_fixed.onnx"
    
    logger.info(f"Saving fixed model: {output_path}")
    onnx.save(model, str(output_path))
    
    # Verify
    try:
        onnx.checker.check_model(model)
        logger.info("✓ Fixed model passes ONNX checker")
    except Exception as e:
        logger.warning(f"Fixed model checker warning: {e}")
    
    return output_path


if __name__ == '__main__':
    # Test on mobilenet_v2
    onnx_dir = Path(r"D:\Base-dir\onnx_models")
    model_name = 'mobilenet_v2'
    
    onnx_path = onnx_dir / model_name / f"{model_name}.onnx"
    
    logger.info("="*70)
    logger.info("ONNX Clip Operator Fixer")
    logger.info("="*70)
    
    if onnx_path.exists():
        fixed_path = fix_onnx_for_onnx2keras(onnx_path)
        logger.info(f"\n✓ Fixed model saved to: {fixed_path}")
    else:
        logger.error(f"ONNX file not found: {onnx_path}")
