"""Neural Genetic Language Interpreter for PyTorch with fault tolerance and flexible parameter handling."""

import torch
import numpy as np
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any

# Configure logger for this module
logger = logging.getLogger(__name__)

class GeneticInterpreter:
    """Translates genetic language instructions into PyTorch operations with fault tolerance."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.registers: Dict[int, torch.Tensor] = {}
        self.special_registers: Dict[int, torch.Tensor] = {}
        self.instruction_pointer: int = 0
        self.program: List[List] = []
        self.debug: bool = False
        self.log_execution: bool = True  # Enable detailed execution logging
        
        # Execution statistics
        self.execution_stats = {
            "total_instructions": 0,
            "successful_instructions": 0,
            "failed_instructions": 0,
            "failure_types": {}
        }
        
        # Default configuration
        self.config = {
            'max_computation_factor': 10,  # Multiplier for program length
            'min_computation_budget': 1000,  # Minimum computation budget
            'fault_tolerant': True,  # Enable fault tolerance (skip failing instructions)
            'max_failure_ratio': 0.5,  # Max ratio of failures before declaring program invalid
            **(config or {})
        }
    
    def initialize(self, inputs: Optional[Dict] = None) -> None:
        """Initialize the interpreter with input tensors."""
        self.registers.clear()
        self.special_registers.clear()
        self.instruction_pointer = 0
        
        # Reset execution statistics
        self.execution_stats = {
            "total_instructions": 0,
            "successful_instructions": 0,
            "failed_instructions": 0,
            "failure_types": {}
        }
        
        inputs = inputs or {}
        if 'y_true' in inputs:
            self.special_registers[0] = inputs['y_true']
            logger.debug(f"Initialized special register 0 (y_true): shape={inputs['y_true'].shape}, "
                        f"requires_grad={inputs['y_true'].requires_grad}")
        if 'y_pred' in inputs:
            self.special_registers[1] = inputs['y_pred']
            logger.debug(f"Initialized special register 1 (y_pred): shape={inputs['y_pred'].shape}, "
                        f"requires_grad={inputs['y_pred'].requires_grad}")
            
        # Initialize other special registers
        self.special_registers[4] = torch.tensor(1e-7)  # EPSILON
        
        if 'y_true' in inputs:
            shape = inputs['y_true'].shape
            self.special_registers[5] = torch.ones(shape)  # ONE
            self.special_registers[6] = torch.zeros(shape)  # ZERO
            
        if 'y_true' in inputs and len(inputs['y_true'].shape) > 0:
            self.special_registers[2] = torch.tensor(inputs['y_true'].shape[0])  # BATCH_SIZE
            
        self.special_registers[3] = torch.tensor(0.001)  # LEARNING_RATE
        
        # Log all special registers
        if self.log_execution:
            for idx, tensor in self.special_registers.items():
                name_map = {0: "y_true", 1: "y_pred", 2: "BATCH_SIZE", 
                           3: "LEARNING_RATE", 4: "EPSILON", 5: "ONE", 6: "ZERO"}
                name = name_map.get(idx, f"special_{idx}")
                try:
                    logger.debug(f"Special register {idx} ({name}): "
                                f"shape={tensor.shape if hasattr(tensor, 'shape') else 'scalar'}, "
                                f"requires_grad={tensor.requires_grad if hasattr(tensor, 'requires_grad') else 'N/A'}")
                except Exception as e:
                    logger.debug(f"Error logging special register {idx}: {e}")
    
    def load_program(self, program: List[List]) -> None:
        """Load a genetic program."""
        self.program = program
        self.instruction_pointer = 0
        
        if self.log_execution:
            logger.debug(f"Loaded program with {len(program)} instructions")
            
            # Log operation code distribution
            op_codes = {}
            for instruction in program:
                if not isinstance(instruction, list) or len(instruction) == 0:
                    continue
                    
                op_code = instruction[0]
                op_codes[op_code] = op_codes.get(op_code, 0) + 1
                
            logger.debug(f"Operation code distribution: {op_codes}")
            
            # Check for potentially problematic operation codes
            for op_code in op_codes:
                if 200 <= op_code < 250 and op_code > 203 and op_code != 205:
                    logger.debug(f"Program contains unsupported reduction operation code: {op_code}")
    
    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        self.debug = enabled
    
    def execute(self) -> torch.Tensor:
        """
        Execute the loaded program with fault tolerance.
        
        Returns:
            The result tensor or a default tensor if execution failed completely.
        """
        if not self.program:
            raise ValueError("No program loaded")
            
        # Initialize computation budget based on program size and configuration
        computation_budget = max(
            self.config['min_computation_budget'], 
            len(self.program) * self.config['max_computation_factor']
        )
        computation_used = 0
        
        while self.instruction_pointer < len(self.program) and computation_used < computation_budget:
            # Increment computation used and stats counter
            computation_used += 1
            self.execution_stats["total_instructions"] += 1
            
            instruction = self.program[self.instruction_pointer]
            if not isinstance(instruction, list):
                logger.warning(f"Invalid instruction at position {self.instruction_pointer}: {instruction}. Skipping.")
                self.instruction_pointer += 1
                self.execution_stats["failed_instructions"] += 1
                self._record_failure("invalid_instruction_format")
                continue
                
            if len(instruction) == 0:
                logger.warning(f"Empty instruction at position {self.instruction_pointer}. Skipping.")
                self.instruction_pointer += 1
                self.execution_stats["failed_instructions"] += 1
                self._record_failure("empty_instruction")
                continue
                
            op_code = instruction[0]
            if self.debug:
                print(f"Executing {instruction} at {self.instruction_pointer}")
                
            if self.log_execution:
                logger.debug(f"Executing instruction {self.instruction_pointer}: op_code={op_code}, params={instruction[1:]}")
                
            # Execute with fault tolerance
            if op_code == 453:                        # RETURN
                reg_idx = instruction[1]
                try:
                    reg_idx = int(reg_idx)            # coerce anything to int
                except (TypeError, ValueError):
                    reg_idx = 9                       # safe fallback
                result = self.registers.get(reg_idx, torch.tensor(0.0))
                return result
            try:
                self._execute_instruction(op_code, instruction[1:])
                self.execution_stats["successful_instructions"] += 1
            except Exception as e:
                self.execution_stats["failed_instructions"] += 1
                error_type = type(e).__name__
                self._record_failure(error_type)
                
                if self.config.get('fault_tolerant', True):
                    logger.warning(f"Error executing instruction {instruction} at position {self.instruction_pointer}: {e}")
                    if self.log_execution:
                        logger.debug(f"Instruction execution error details: {traceback.format_exc()}")
                    # Continue execution with next instruction
                else:
                    # Non-fault-tolerant mode: re-raise the exception
                    logger.error(f"Error executing instruction {instruction} at position {self.instruction_pointer}: {e}")
                    logger.error(traceback.format_exc())
                    raise
                    
            self.instruction_pointer += 1
        
        # Check if execution was terminated due to budget exhaustion
        if computation_used >= computation_budget:
            logger.debug(f"Program execution terminated: exceeded computation budget ({computation_budget})")
            
        # Check if the program had too many failures
        failure_ratio = self.execution_stats["failed_instructions"] / max(1, self.execution_stats["total_instructions"])
        if failure_ratio > self.config.get('max_failure_ratio', 0.5):
            logger.warning(f"Program had high failure ratio: {failure_ratio:.2f} ({self.execution_stats['failed_instructions']}/{self.execution_stats['total_instructions']} instructions failed)")
            
        
        
        if self.log_execution:
            try:
                logger.debug(f"Program execution complete. Result: {result.item():.6f}, "
                            f"shape={result.shape}, requires_grad={result.requires_grad}")
                logger.debug(f"Execution stats: {self.execution_stats}")
            except Exception as e:
                logger.debug(f"Error logging result: {e}")
                
        return result
    
    def _record_failure(self, error_type: str) -> None:
        """Record a failure type in execution statistics."""
        self.execution_stats["failure_types"][error_type] = self.execution_stats["failure_types"].get(error_type, 0) + 1
    
    def _execute_instruction(self, op_code: int, params: List) -> None:
        """Execute a single instruction."""
        # Arithmetic Operations (000-099)
        if 0 <= op_code < 100:
            self._execute_arithmetic_op(op_code, params)
        # Tensor Operations (100-199)
        elif 100 <= op_code < 200:
            self._execute_tensor_op(op_code, params)
        # Reduction Operations (200-249)
        elif 200 <= op_code < 250:
            self._execute_reduction_op(op_code, params)
        # Neural Network Operations (250-349)
        elif 250 <= op_code < 350:
            self._execute_nn_op(op_code, params)
        # Loss Function Components (350-399)
        elif 350 <= op_code < 400:
            self._execute_loss_op(op_code, params)
        # Logical & Comparison Operations (400-449)
        elif 400 <= op_code < 450:
            self._execute_logical_op(op_code, params)
        # Control Flow Operations (450-499)
        elif 450 <= op_code < 500:
            self._execute_control_flow_op(op_code, params)
        else:
            logger.warning(f"Unknown operation code: {op_code}")
            raise ValueError(f"Unknown operation code: {op_code}")
    
    def _get_register_value(self, reg: int) -> torch.Tensor:
        """Get the value from a register."""
        try:
            reg = int(reg)  # Convert to int if it's another type
            if 0 <= reg < 20:  # General register
                return self.registers.get(reg, torch.tensor(0.0))
            elif 100 <= reg < 110:  # Special register
                return self.special_registers.get(reg - 100, torch.tensor(0.0))
            else:
                logger.warning(f"Invalid register: {reg}")
                return torch.tensor(0.0)  # Return default value for invalid registers
        except (TypeError, ValueError):
            logger.warning(f"Invalid register reference: {reg}")
            return torch.tensor(0.0)  # Return default value for non-integer registers
    
    def _set_register_value(self, reg: int, value: torch.Tensor) -> None:
        """Set the value in a register."""
        try:
            reg = int(reg)  # Convert to int if it's another type
            if 0 <= reg < 20:
                self.registers[reg] = value
                
                if self.log_execution:
                    try:
                        logger.debug(f"Set register {reg} to tensor with shape={value.shape if hasattr(value, 'shape') else 'scalar'}, "
                                    f"requires_grad={value.requires_grad if hasattr(value, 'requires_grad') else 'N/A'}")
                    except Exception as e:
                        logger.debug(f"Error logging register {reg}: {e}")
            else:
                logger.warning(f"Cannot write to invalid register: {reg}")
                # Silently fail for invalid registers
        except (TypeError, ValueError):
            logger.warning(f"Cannot write to non-integer register: {reg}")
            # Silently fail for non-integer registers
    
    def _execute_arithmetic_op(self, op_code: int, params: List) -> None:
        """Execute arithmetic operations with flexible parameter handling."""
        try:
            if op_code == 1:  # ADD
                if len(params) < 3:
                    logger.warning(f"ADD operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 1
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                self._set_register_value(rz, x + y)
            elif op_code == 2:  # SUB
                if len(params) < 3:
                    logger.warning(f"SUB operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 1
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                self._set_register_value(rz, x - y)
            elif op_code == 3:  # MUL
                if len(params) < 3:
                    logger.warning(f"MUL operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 1
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                self._set_register_value(rz, x * y)
            elif op_code == 4:  # DIV
                if len(params) < 3:
                    logger.warning(f"DIV operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 1
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                epsilon = self.special_registers.get(4, torch.tensor(1e-7))
                # Add epsilon to avoid division by zero
                y_safe = y + epsilon
                self._set_register_value(rz, x / y_safe)
            elif op_code == 5:  # POW
                if len(params) < 3:
                    logger.warning(f"POW operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 1
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                self._set_register_value(rz, x ** y)
            elif op_code == 6:  # SQRT
                if len(params) < 2:
                    logger.warning(f"SQRT operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only, ignore extras
                x = self._get_register_value(rx)
                # Ensure x is non-negative
                x_safe = torch.clamp(x, min=0.0)
                self._set_register_value(rz, torch.sqrt(x_safe))
            elif op_code == 7:  # LOG
                if len(params) < 2:
                    logger.warning(f"LOG operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                epsilon = self.special_registers.get(4, torch.tensor(1e-7))
                # Add epsilon to avoid log of zero or negative
                x_safe = torch.clamp(x, min=epsilon)
                self._set_register_value(rz, torch.log(x_safe))
            elif op_code == 8:  # EXP
                if len(params) < 2:
                    logger.warning(f"EXP operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                # Clamp to avoid overflow
                x_safe = torch.clamp(x, max=88.0)  # Exp(88) is near the max value for float32
                self._set_register_value(rz, torch.exp(x_safe))
            elif op_code == 9:  # ABS
                if len(params) < 2:
                    logger.warning(f"ABS operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                self._set_register_value(rz, torch.abs(x))
            elif op_code == 10:  # SQUARE
                if len(params) < 2:
                    logger.warning(f"SQUARE operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                self._set_register_value(rz, x ** 2)
            elif op_code == 11:  # NEGATE
                if len(params) < 2:
                    logger.warning(f"NEGATE operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                self._set_register_value(rz, -x)
            elif op_code == 12:  # RECIPROCAL
                if len(params) < 2:
                    logger.warning(f"RECIPROCAL operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                epsilon = self.special_registers.get(4, torch.tensor(1e-7))
                # Add epsilon to avoid division by zero
                x_safe = x + epsilon
                self._set_register_value(rz, 1.0 / x_safe)
            elif op_code == 20:  # ADD_CONST
                if len(params) < 3:
                    logger.warning(f"ADD_CONST operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    c = params[1] if len(params) > 1 else 1.0
                    rz = 9  # Default output register
                else:
                    rx, c, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                self._set_register_value(rz, x + c)
            elif op_code == 21:  # MUL_CONST
                if len(params) < 3:
                    logger.warning(f"MUL_CONST operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    c = params[1] if len(params) > 1 else 1.0
                    rz = 9  # Default output register
                else:
                    rx, c, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                self._set_register_value(rz, x * c)
            else:
                logger.warning(f"Unsupported arithmetic operation: {op_code}")
                raise ValueError(f"Unsupported arithmetic operation: {op_code}")
        except Exception as e:
            logger.warning(f"Error in arithmetic operation {op_code}: {e}")
            raise
    
    def _execute_tensor_op(self, op_code: int, params: List) -> None:
        """Execute tensor operations with flexible parameter handling."""
        try:
            if op_code == 100:  # LOAD from special register
                if len(params) < 2:
                    logger.warning(f"LOAD operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    sy = 0  # Default to y_true special register
                else:
                    rx, sy = params[0], params[1]  # Take first 2 params only
                value = self.special_registers.get(sy, torch.tensor(0.0))
                self._set_register_value(rx, value)
            elif op_code == 101:  # LOAD_CONST
                if len(params) < 2:
                    logger.warning(f"LOAD_CONST operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    c = 1.0  # Default constant
                else:
                    rx, c = params[0], params[1]  # Take first 2 params only
                self._set_register_value(rx, torch.tensor(c))
            elif op_code == 102:  # MATMUL
                if len(params) < 3:
                    logger.warning(f"MATMUL operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 2
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                try:
                    self._set_register_value(rz, torch.matmul(x, y))
                except RuntimeError as e:
                    logger.warning(f"MATMUL operation failed: {e}. Using element-wise multiply instead.")
                    self._set_register_value(rz, x * y)
            elif op_code == 103:  # TRANSPOSE
                if len(params) < 2:
                    logger.warning(f"TRANSPOSE operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                try:
                    if x.dim() >= 2:
                        self._set_register_value(rz, x.transpose(0, 1))
                    else:
                        # Can't transpose scalars or 1D tensors
                        self._set_register_value(rz, x)
                except RuntimeError as e:
                    logger.warning(f"TRANSPOSE operation failed: {e}. Passing through original tensor.")
                    self._set_register_value(rz, x)
            elif op_code == 110:  # CLONE
                if len(params) < 2:
                    logger.warning(f"CLONE operation requires 2 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    rz = 9  # Default output register
                else:
                    rx, rz = params[0], params[1]  # Take first 2 params only
                x = self._get_register_value(rx)
                self._set_register_value(rz, x.clone())
            else:
                logger.warning(f"Unsupported tensor operation: {op_code}")
                raise ValueError(f"Unsupported tensor operation: {op_code}")
        except Exception as e:
            logger.warning(f"Error in tensor operation {op_code}: {e}")
            raise
    
    def _execute_reduction_op(self, op_code: int, params: List) -> None:
        """Execute reduction operations with flexible parameter handling."""
        try:
            if len(params) < 3:
                logger.warning(f"Reduction operation requires at least 3 parameters, got {len(params)}. Using defaults.")
                rx = params[0] if len(params) > 0 else 1
                axis = params[1] if len(params) > 1 else -1  # Default to last dimension
                rz = 9  # Default output register
            else:
                rx, axis, rz = params[0], params[1], params[2]  # Take first 3 params only
                
            x = self._get_register_value(rx)
            
            # Handle scalar inputs (0-dim tensors) gracefully
            if x.dim() == 0:
                logger.warning(f"Attempted reduction on scalar tensor with axis {axis}. Passing through original value.")
                self._set_register_value(rz, x)
                return
                
            # Handle negative axis - wrap around
            if axis < 0:
                axis = x.dim() + axis
            
            # Handle axis out of bounds
            if axis >= x.dim():
                logger.warning(f"Reduction axis {axis} out of bounds for tensor with {x.dim()} dimensions. Using axis 0 instead.")
                axis = 0 if x.dim() > 0 else None
                
            if op_code == 200:  # REDUCE_SUM
                result = torch.sum(x, dim=axis)
            elif op_code == 201:  # REDUCE_MEAN
                result = torch.mean(x, dim=axis)
            elif op_code == 202:  # REDUCE_MAX
                # Handle tuple return value
                max_result = torch.max(x, dim=axis)
                result = max_result.values  # Extract just the values
            elif op_code == 203:  # REDUCE_MIN
                # Handle tuple return value
                min_result = torch.min(x, dim=axis)
                result = min_result.values  # Extract just the values
            elif op_code == 205:  # REDUCE_STD
                result = torch.std(x, dim=axis)
            else:
                logger.warning(f"Unsupported reduction operation: {op_code}")
                # Fallback to mean reduction for unknown reduction ops
                logger.warning(f"Falling back to REDUCE_MEAN for unknown reduction op {op_code}")
                result = torch.mean(x, dim=axis)
                
            self._set_register_value(rz, result)
                
        except RuntimeError as e:
            logger.warning(f"Runtime error in reduction operation: {e}. Passing through input tensor.")
            self._set_register_value(rz, x)
        except Exception as e:
            logger.warning(f"Error in reduction operation {op_code}: {e}")
            raise
    
    def _execute_nn_op(self, op_code: int, params: List) -> None:
        """Execute neural network operations with flexible parameter handling."""
        try:
            # Get parameters with defaults
            rx = params[0] if len(params) > 0 else 1
            ry = params[1] if len(params) > 1 else 2
            rz = params[2] if len(params) > 2 else 9  # Default output register
            
            x = self._get_register_value(rx)
            
            if op_code == 300:  # MAX operation
                y = self._get_register_value(ry)
                result = torch.maximum(x, y)
                self._set_register_value(rz, result)
                return
            x = self._get_register_value(rx)
            
            if op_code == 250:  # RELU
                result = torch.relu(x)
            elif op_code == 251:  # SIGMOID
                result = torch.sigmoid(x)
            elif op_code == 252:  # TANH
                result = torch.tanh(x)
            elif op_code == 253:  # SOFTMAX
                if x.dim() == 0:
                    # Can't apply softmax to scalars
                    result = x
                else:
                    # Use last dimension for softmax
                    result = torch.softmax(x, dim=-1)
            else:
                logger.warning(f"Unsupported neural network operation: {op_code}")
                # Default to identity function
                result = x
                
            self._set_register_value(rz, result)
                
        except RuntimeError as e:
            logger.warning(f"Runtime error in neural network operation: {e}. Passing through input tensor.")
            self._set_register_value(rz, x)
        except Exception as e:
            logger.warning(f"Error in neural network operation {op_code}: {e}")
            raise
    
    def _execute_loss_op(self, op_code: int, params: List) -> None:
        """Execute loss function operations with flexible parameter handling."""
        try:
            if len(params) < 3:
                logger.warning(f"Loss operation requires 3 parameters, got {len(params)}. Using defaults.")
                rx = params[0] if len(params) > 0 else 1
                ry = params[1] if len(params) > 1 else 2
                rz = 9  # Default output register
            else:
                rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params only
                
            y_true = self._get_register_value(rx)
            y_pred = self._get_register_value(ry)
            
            # Check if shapes match, broadcast if possible
            if y_true.shape != y_pred.shape:
                try:
                    # Try broadcasting
                    _ = y_true + y_pred  # This will raise an error if not broadcastable
                except RuntimeError:
                    logger.warning(f"Shape mismatch in loss operation: y_true {y_true.shape} vs y_pred {y_pred.shape}. Attempting to reshape.")
                    # Try to reshape one to match the other
                    try:
                        if y_true.numel() == y_pred.numel():
                            y_true = y_true.reshape(y_pred.shape)
                        else:
                            logger.warning("Shapes not compatible. Using element-wise ops where possible.")
                    except RuntimeError:
                        logger.warning("Could not reshape tensors to match. Using element-wise ops where possible.")
            
            if op_code == 350:  # MSE
                diff = y_true - y_pred
                squared_diff = diff ** 2
                result = torch.mean(squared_diff)
            elif op_code == 351:  # MAE
                diff = y_true - y_pred
                abs_diff = torch.abs(diff)
                result = torch.mean(abs_diff)
            elif op_code == 352:  # BCE
                epsilon = self.special_registers.get(4, torch.tensor(1e-7))
                # Ensure y_pred is in valid range for BCE
                clipped = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
                log_pred = torch.log(clipped)
                log_one_minus = torch.log(1.0 - clipped)
                first_term = y_true * log_pred
                second_term = (1.0 - y_true) * log_one_minus
                loss = -(first_term + second_term)
                result = torch.mean(loss)
            else:
                logger.warning(f"Unsupported loss operation: {op_code}")
                # Fallback to MSE for unknown loss ops
                logger.warning(f"Falling back to MSE for unknown loss op {op_code}")
                diff = y_true - y_pred
                squared_diff = diff ** 2
                result = torch.mean(squared_diff)
                
            self._set_register_value(rz, result)
                
        except RuntimeError as e:
            logger.warning(f"Runtime error in loss operation: {e}. Setting default loss value.")
            # Use a default loss value
            self._set_register_value(rz, torch.tensor(1.0, requires_grad=True))
        except Exception as e:
            logger.warning(f"Error in loss operation {op_code}: {e}")
            raise
    
    def _execute_logical_op(self, op_code: int, params: List) -> None:
        """Execute logical operations with flexible parameter handling."""
        try:
            if op_code == 400:  # GT
                if len(params) < 3:
                    logger.warning(f"GT operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 2
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                self._set_register_value(rz, x > y)
            elif op_code == 401:  # GE
                if len(params) < 3:
                    logger.warning(f"GE operation requires 3 parameters, got {len(params)}. Using defaults.")
                    rx = params[0] if len(params) > 0 else 1
                    ry = params[1] if len(params) > 1 else 2
                    rz = 9  # Default output register
                else:
                    rx, ry, rz = params[0], params[1], params[2]  # Take first 3 params
                x = self._get_register_value(rx)
                y = self._get_register_value(ry)
                self._set_register_value(rz, x >= y)
            elif op_code == 409:  # WHERE
                if len(params) < 4:
                    logger.warning(f"WHERE operation requires 4 parameters, got {len(params)}. Using defaults.")
                    raise
                else:
                    rx, ry, rz, relse = params[0], params[1], params[2], params[3]  # Take first 4 params
                
                    x = self._get_register_value(rx)  # Assume x is a tensor
                    y = self._get_register_value(ry)  # Assume y is a tensor

                    else_val = self._get_register_value(relse) 

                try:
                    # Replace if-else with torch.where
                    result = torch.where(x >= y, 
                                        x,  # What you want when x >= y
                                        else_val) # What you want when x < y
                    self._set_register_value(rz, result)
                except RuntimeError as e:
                    logger.warning(f"WHERE operation failed: {e}. Using x value.")
                    #self._set_register_value(rz, x)
            else:
                logger.warning(f"Unsupported logical operation: {op_code}")
                raise ValueError(f"Unsupported logical operation: {op_code}")
        except Exception as e:
            logger.warning(f"Error in logical operation {op_code}: {e}")
            raise
    
    def _execute_control_flow_op(self, op_code: int, params: List) -> None:
        """Execute control flow operations with flexible parameter handling."""
        try:
            if op_code == 450:  # JUMP
                if len(params) < 1:
                    logger.warning(f"JUMP operation requires 1 parameter, got {len(params)}. Ignoring jump.")
                    return
                addr = params[0]
                # Validate jump target
                if not (0 <= addr < len(self.program)):
                    logger.warning(f"Invalid jump target: {addr}, max allowed: {len(self.program)-1}. Ignoring jump.")
                    return
                self.instruction_pointer = addr - 1  # -1 because it will be incremented
            elif op_code == 451:  # JUMP_IF
                if len(params) < 2:
                    logger.warning(f"JUMP_IF operation requires 2 parameters, got {len(params)}. Ignoring jump.")
                    return
                rcond, addr = params[0], params[1]
                # Validate jump target
                if not (0 <= addr < len(self.program)):
                    logger.warning(f"Invalid jump target: {addr}, max allowed: {len(self.program)-1}. Ignoring jump.")
                    return
                condition = self._get_register_value(rcond)
                jump_taken = torch.any(condition != 0)
                if jump_taken:
                    self.instruction_pointer = addr - 1
            elif op_code == 454:  # END
                self.instruction_pointer = len(self.program)  # End execution
            else:
                logger.warning(f"Unsupported control flow operation: {op_code}")
                raise ValueError(f"Unsupported control flow operation: {op_code}")
        except Exception as e:
            logger.warning(f"Error in control flow operation: {e}")
            raise
    
    def dispose(self) -> None:
        """Clean up resources."""
        self.registers.clear()
        self.special_registers.clear()
    
    @staticmethod
    def decode_to_function(genetic_code: List[List], config: Optional[Dict] = None) -> callable:
        """Convert genetic code to a Python function."""
        logger.debug(f"Decoding genetic code to function, code length: {len(genetic_code)}")
        
        # Ensure the config has fault_tolerant set to True if not specified
        if config is None:
            config = {'fault_tolerant': True}
        elif 'fault_tolerant' not in config:
            config['fault_tolerant'] = True
        
        def loss_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            try:
                interpreter = GeneticInterpreter(config)
                interpreter.initialize({'y_true': y_true, 'y_pred': y_pred})
                interpreter.load_program(genetic_code)
                result = interpreter.execute()
                
                # Ensure the result is a scalar
                if result.numel() != 1:
                    logger.warning(f"Loss function returned non-scalar result with shape {result.shape}. Taking mean.")
                    result = torch.mean(result)
                
                # Ensure the result requires gradient
                if not result.requires_grad and y_pred.requires_grad:
                    logger.warning("Loss result doesn't require grad but y_pred does. Creating a differentiable path.")
                    # Create a differentiable path by multiplying by a dummy variable
                    dummy = torch.ones(1, requires_grad=True)
                    result = result * dummy
                
                return result
            except Exception as e:
                logger.error(f"Error in loss function: {e}")
                logger.error(traceback.format_exc())
                
                # Return a default loss value that requires gradient
                default_loss = torch.tensor(1.0, requires_grad=True)
                return default_loss
                
        return loss_fn
