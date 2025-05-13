"""Genetic Algorithm Strategies for AutoML."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import random
import numpy as np
import copy

class GeneticAlgorithmStrategy(ABC):
    """Interface for genetic algorithm strategies."""
    
    @abstractmethod
    def create_initial_population(self, seed: List, population_size: int, 
                                mutation_rate: float) -> List[List]:
        """Create initial population from seed function."""
        pass
        
    @abstractmethod
    def evolve_population(self, population: List, fitnesses: np.ndarray, 
                        config: Dict) -> List[List]:
        """Evolve population based on fitness scores."""
        pass
        
    @abstractmethod
    def clone_function(self, function: List) -> List:
        """Create a deep copy of a function."""
        pass

    def _generate_param(self, param_type: str) -> Any:
        """Generate a parameter value based on its type."""
        if param_type == "reg":
            return random.randint(1, 9)  # General register (1-9)
        elif param_type == "special_reg":
            return random.randint(0, 6)  # Special register index
        elif param_type == "axis":
            return random.choice([-1, 0])  # Common axis values
        elif param_type == "addr":
            return random.randint(0, 10)  # Instruction address
        elif param_type == "const":
            return random.random() * 2 - 1  # Random value between -1 and 1
        else:
            return 0  # Default
        
    def _adapt_param_value(self, param: Any, param_type: str) -> Any:
        """Adapt a parameter value to ensure it's appropriate for the specified type."""
        if param_type == "reg":
            # Ensure register is in valid range (0-19), preferring 1-9 for usability
            try:
                reg_value = int(param)
            except (ValueError, TypeError):
                reg_value = random.randint(1, 9)
            return max(0, min(19, reg_value))
        elif param_type == "special_reg":
            try:
                sreg_value = int(param)
            except (ValueError, TypeError):
                sreg_value = random.randint(0, 6)
            return max(0, min(6, sreg_value))
        elif param_type == "axis":
            # Use -1 for last dimension or 0 for first dimension
            if isinstance(param, (int, float)):
                return -1 if param < 0 else 0
            return -1  # Default to last dimension
        elif param_type == "addr":
            try:
                addr_value = int(param)
            except (ValueError, TypeError):
                addr_value = random.randint(0, 10)
            return max(0, addr_value)  # No upper limit for addresses
        elif param_type == "const":
            if isinstance(param, (int, float)):
                return max(-100, min(100, param))  # Clamp extreme values
            return random.random() * 2 - 1  # Default to random in [-1, 1]
        else:
            return param  # Return as is for unknown types
    
    def _mutate_parameters(self, func: List, index: int) -> None:
        """Mutate the parameters of an instruction."""
        if index >= len(func):
            return
            
        instr = func[index]
        if len(instr) <= 1:
            return
            
        op_code = instr[0]
        op_spec = self.op_registry.get(op_code)
        
        if not op_spec:
            return  # Skip if operation not in registry
            
        # Pick a random parameter to mutate (excluding op code)
        if len(instr) > 1:
            param_idx = random.randint(1, len(instr) - 1)
            
            # Get the parameter type
            if param_idx - 1 < len(op_spec.get("param_types", [])):
                param_type = op_spec["param_types"][param_idx - 1]
            else:
                # Default to register type if index exceeds defined types
                param_type = "reg"
            
            # Generate a new value based on type and adapt it
            new_value = self._generate_param(param_type)
            adapted_value = self._adapt_param_value(new_value, param_type)
            
            instr[param_idx] = adapted_value
    
    def _insert_instruction(self, func: List, index: int) -> None:
        """Insert a new random instruction."""
        # Select a random operation type
        categories = list(self.op_categories.keys())
        category = random.choice(categories)
        if category == "control":
            category = "arithmetic"  # Avoid inserting control flow operations randomly
            
        op_code = random.choice(self.op_categories[category])
        op_spec = self.op_registry.get(op_code)
        
        if not op_spec:
            return  # Skip if operation not in registry
            
        # Create a new instruction with the operation code
        new_instr = [op_code]
        
        # Add appropriate parameters
        for param_type in op_spec["param_types"]:
            param_value = self._generate_param(param_type)
            adapted_value = self._adapt_param_value(param_value, param_type)
            new_instr.append(adapted_value)
            
        # Insert the new instruction
        func.insert(index, new_instr)
    
    def _delete_instruction(self, func: List, index: int) -> None:
        """Delete an instruction if it's safe to do so."""
        if len(func) <= 2 or index >= len(func):
            return  # Keep at least 2 instructions
            
        # Don't delete RETURN or critical setup instructions
        if index == len(func) - 1 and func[index][0] == 453:
            return
        if index < 2 and func[index][0] == 100:  # LOAD operations at the start
            return
            
        func.pop(index)
    
    def _swap_with_non_critical(self, func: List, index: int) -> None:
        """Swap an instruction with another non-critical instruction."""
        if len(func) <= 3:  # Not enough instructions to swap safely
            return
            
        # Find candidate indices (skip first two and last one)
        candidates = [i for i in range(2, len(func) - 1) if i != index]
        if not candidates:
            return
            
        other_idx = random.choice(candidates)
        func[index], func[other_idx] = func[other_idx], func[index]

    def _fix_params(self, instr: List) -> None:
        """
        Ensure instr[1:] has exactly the right number of parameters
        for its opcode, according to self.op_registry.
        """
        op_code = instr[0]
        spec = self.op_registry.get(op_code)
        if spec is None:
            logger.warning(f"Unknown opcode {op_code}; leaving params untouched")
            return
        needed = spec["params"]
        types  = spec["param_types"]

        # drop extras
        params = instr[1:1+needed]

        # pad missing
        while len(params) < needed:
            ptype = types[len(params)]
            raw    = self._generate_param(ptype)
            adapted= self._adapt_param_value(raw, ptype)
            params.append(adapted)

        instr[1:] = params

    def _load_operation_registry(self) -> Dict[int, Dict]:
        """Load the operation registry with parameter specifications."""
        return {
            # Arithmetic Operations (0-99)
            1: {"name": "ADD", "params": 3, "param_types": ["reg", "reg", "reg"]},
            2: {"name": "SUB", "params": 3, "param_types": ["reg", "reg", "reg"]},
            3: {"name": "MUL", "params": 3, "param_types": ["reg", "reg", "reg"]},
            4: {"name": "DIV", "params": 3, "param_types": ["reg", "reg", "reg"]},
            5: {"name": "POW", "params": 3, "param_types": ["reg", "reg", "reg"]},
            6: {"name": "SQRT", "params": 2, "param_types": ["reg", "reg"]},
            7: {"name": "LOG", "params": 2, "param_types": ["reg", "reg"]},
            8: {"name": "EXP", "params": 2, "param_types": ["reg", "reg"]},
            9: {"name": "ABS", "params": 2, "param_types": ["reg", "reg"]},
            10: {"name": "SQUARE", "params": 2, "param_types": ["reg", "reg"]},
            11: {"name": "NEGATE", "params": 2, "param_types": ["reg", "reg"]},
            12: {"name": "RECIPROCAL", "params": 2, "param_types": ["reg", "reg"]},
            20: {"name": "ADD_CONST", "params": 3, "param_types": ["reg", "const", "reg"]},
            21: {"name": "MUL_CONST", "params": 3, "param_types": ["reg", "const", "reg"]},
            
            # Tensor Operations (100-199)
            100: {"name": "LOAD", "params": 2, "param_types": ["reg", "special_reg"]},
            101: {"name": "LOAD_CONST", "params": 2, "param_types": ["reg", "const"]},
            102: {"name": "MATMUL", "params": 3, "param_types": ["reg", "reg", "reg"]},
            103: {"name": "TRANSPOSE", "params": 2, "param_types": ["reg", "reg"]},
            110: {"name": "CLONE", "params": 2, "param_types": ["reg", "reg"]},
            
            # Reduction Operations (200-249)
            200: {"name": "REDUCE_SUM", "params": 3, "param_types": ["reg", "axis", "reg"]},
            201: {"name": "REDUCE_MEAN", "params": 3, "param_types": ["reg", "axis", "reg"]},
            202: {"name": "REDUCE_MAX", "params": 3, "param_types": ["reg", "axis", "reg"]},
            203: {"name": "REDUCE_MIN", "params": 3, "param_types": ["reg", "axis", "reg"]},
            
            # Neural Network Operations (250-349)
            250: {"name": "RELU", "params": 2, "param_types": ["reg", "reg"]},
            251: {"name": "SIGMOID", "params": 2, "param_types": ["reg", "reg"]},
            252: {"name": "TANH", "params": 2, "param_types": ["reg", "reg"]},
            253: {"name": "SOFTMAX", "params": 2, "param_types": ["reg", "reg"]},
            300: {"name": "MAX", "params": 3, "param_types": ["reg", "reg", "reg"]},
            
            # Loss Function Operations (350-399)
            350: {"name": "MSE", "params": 3, "param_types": ["reg", "reg", "reg"]},
            351: {"name": "MAE", "params": 3, "param_types": ["reg", "reg", "reg"]},
            352: {"name": "BCE", "params": 3, "param_types": ["reg", "reg", "reg"]},
            
            # Logical Operations (400-449)
            400: {"name": "GT", "params": 3, "param_types": ["reg", "reg", "reg"]},
            401: {"name": "GE", "params": 3, "param_types": ["reg", "reg", "reg"]},
            409: {"name": "WHERE", "params": 4, "param_types": ["reg", "reg", "reg", "reg"]},
            
            # Control Flow Operations (450-499)
            450: {"name": "JUMP", "params": 1, "param_types": ["addr"]},
            451: {"name": "JUMP_IF", "params": 2, "param_types": ["reg", "addr"]},
            453: {"name": "RETURN", "params": 1, "param_types": ["reg"]},
            454: {"name": "END", "params": 0, "param_types": []}
        }
        
    def _load_operation_categories(self) -> Dict[str, List[int]]:
        """Load operation categories for weighted selection."""
        return {
            "arithmetic": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21],
            "tensor": [100, 101, 102, 103, 110],
            "reduction": [200, 201, 202, 203],
            "nn": [250, 251, 252, 253],
            "loss": [350, 351, 352],
            "logical": [400, 401, 409],
            "control": [450, 451, 453, 454]
        }


import random
import numpy as np
import copy
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RegisterValidator:
    """Helper class to validate and fix register references in genetic code."""
    
    @staticmethod
    def validate_and_fix_genetic_code(code: List[List]) -> List[List]:
        """
        Validate all register references in the genetic code and fix any invalid ones.
        
        Args:
            code: The genetic code to validate
            
        Returns:
            Fixed genetic code with valid register references
        """
        if not code:
            return code
            
        fixed_code = []
        for instruction in code:
            if not isinstance(instruction, list) or len(instruction) == 0:
                continue
                
            op_code = instruction[0]
            fixed_instruction = [op_code]
            
            # Parameters start at index 1
            for i, param in enumerate(instruction[1:], 1):
                # Check if this parameter is likely a register
                if RegisterValidator._is_likely_register_param(op_code, i):
                    # Fix register value
                    fixed_param = RegisterValidator._fix_register_value(param)
                    fixed_instruction.append(fixed_param)
                else:
                    # Not a register, keep original value
                    fixed_instruction.append(param)
            
            fixed_code.append(fixed_instruction)
            
        # Ensure required structure
        if not fixed_code:
            # Create minimal valid program
            fixed_code = [
                [100, 1, 0],  # LOAD R1, S0 (y_true)
                [100, 2, 1],  # LOAD R2, S1 (y_pred)
                [350, 1, 2, 9],  # MSE R1, R2, R9
                [453, 9]  # RETURN R9
            ]
        else:
            # Ensure first two instructions load the inputs
            if len(fixed_code) >= 1 and fixed_code[0][0] != 100:
                fixed_code[0] = [100, 1, 0]  # LOAD R1, S0 (y_true)
            if len(fixed_code) >= 2 and fixed_code[1][0] != 100:
                fixed_code[1] = [100, 2, 1]  # LOAD R2, S1 (y_pred)
                
            # Ensure last instruction is RETURN with register 9
            if fixed_code[-1][0] != 453:
                fixed_code.append([453, 9])  # RETURN R9
            else:
                fixed_code[-1] = [453, 9]  # Ensure it returns register 9
        
        return fixed_code
    
    @staticmethod
    def _is_likely_register_param(op_code: int, param_index: int) -> bool:
        """
        Determine if a parameter is likely a register based on op code and position.
        
        Args:
            op_code: The operation code
            param_index: The parameter position (1-based)
            
        Returns:
            True if the parameter is likely a register reference
        """
        # This is a simplified heuristic - ideally we'd use the op registry
        if op_code >= 350 and op_code <= 352:  # Loss operations
            return param_index <= 3  # All 3 params are registers
            
        if op_code >= 1 and op_code <= 12:  # Basic arithmetic
            return True  # All params are registers
            
        if op_code == 453:  # Return
            return True  # Single param is register
            
        if 100 <= op_code < 200:  # Tensor ops
            # First param is almost always a register
            if param_index == 1:
                return True
            # For LOAD, second param is special register
            if op_code == 100 and param_index == 2:
                return False
            # For LOAD_CONST, second param is constant
            if op_code == 101 and param_index == 2:
                return False
                
        if 200 <= op_code < 250:  # Reduction ops
            # First and third params are registers, second is axis
            return param_index == 1 or param_index == 3
            
        # Default to treating as register
        return True
    
    @staticmethod
    def _fix_register_value(value: Any) -> int:
        """
        Fix a register value to ensure it's within the valid range (0-19).
        
        Args:
            value: The register value to fix
            
        Returns:
            Fixed register value
        """
        try:
            # Convert to int if possible
            reg_value = int(value)
        except (ValueError, TypeError):
            # Default to register 9 if conversion fails
            reg_value = 9
            
        # Restrict to valid range for general registers
        return max(0, min(19, reg_value))

# Modify StandardGeneticAlgorithm to include register validation
class StandardGeneticAlgorithm(GeneticAlgorithmStrategy):
    """Standard genetic algorithm implementation with register validation."""
    
    def __init__(self, config: Dict = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.op_registry = self._load_operation_registry()
        self.op_categories = self._load_operation_categories()
        

    
    def create_initial_population(self, seed: List, population_size: int = 20, 
                                mutation_rate: float = 0.3) -> List[List]:
        """
        Create a new population based on a seed function.
        
        Args:
            seed: Seed genetic function
            population_size: Size of the population to create
            mutation_rate: Probability of mutation
            
        Returns:
            New population
        """
        # First validate the seed
        validated_seed = RegisterValidator.validate_and_fix_genetic_code(seed)
        population = [self.clone_function(validated_seed)]
        
        for _ in range(1, population_size):
            mutated = self.mutate(
                self.clone_function(validated_seed), 
                mutation_rate
            )
            population.append(mutated)
            
        return population
    
    def clone_function(self, func: List) -> List:
        """Make a deep copy of a genetic function."""
        return copy.deepcopy(func)
    
    def _format_genetic_code(self, code: List) -> str:
        """Format genetic code for logging purposes."""
        op_registry = {v['name']: k for k, v in self.op_registry.items()}
        return '\n'.join(
            f"{self.op_registry.get(instr[0], {}).get('name', 'UNKNOWN'):<12} {instr}" 
            for instr in code
        )

    
    def mutate(self, func: List, mutation_rate: float = 0.3) -> List:
        """
        Mutate a genetic function with various operations.
        
        Args:
            func: Function to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated function
        """
        original_code = self._format_genetic_code(func)
        mutated = self.clone_function(func)
        
        logger.debug(f"Mutating function (original):\n{original_code}")
        
        # Skip the first two instructions (LOAD operations) and the last instruction (RETURN)
        i = 2
        mutations_applied = 0
        while i < len(mutated) - 1:
            if random.random() < mutation_rate:
                original_instruction = str(mutated[i])
                strategy = random.randint(0, 4)
                
                if strategy == 0:  # Modify op code
                    self._flexible_mutate_op_code(mutated, i)
                elif strategy == 1:  # Modify parameters
                    self._mutate_parameters(mutated, i)
                elif strategy == 2 and len(mutated) < 20:  # Insert instruction
                    self._insert_instruction(mutated, i)
                elif strategy == 3 and len(mutated) > 4:  # Delete instruction
                    if i < len(mutated):
                        self._delete_instruction(mutated, i)
                        continue  # Don't increment i since we removed an instruction
                elif strategy == 4:  # Swap instructions
                    self._swap_with_non_critical(mutated, i)
                
                if original_instruction != str(mutated[i]):
                    mutations_applied += 1
                    logger.debug(f"Mutation applied at index {i}:\n" 
                                f"Original: {original_instruction}\n"
                                f"Modified: {mutated[i]}")
            i += 1

        for idx in range(len(mutated)):
            self._fix_params(mutated[idx])

        if mutations_applied > 0:
            logger.debug(f"Applied {mutations_applied} mutations. Final code:\n{self._format_genetic_code(mutated)}")
        
        # Ensure program starts with LOAD instructions
        if len(mutated) >= 2:
            if mutated[0][0] != 100:  # First instruction must be LOAD
                mutated[0] = [100, 1, 0]  # LOAD R1, S0 (y_true)
            if mutated[1][0] != 100:  # Second instruction must be LOAD
                mutated[1] = [100, 2, 1]  # LOAD R2, S1 (y_pred)
        
        # Ensure program ends with a RETURN with register 9
        if not mutated or mutated[-1][0] != 453:
            mutated.append([453, 9])  # RETURN R9
        else:
            mutated[-1] = [453, 9]  # Ensure it returns register 9
        
        # Final validation step to ensure all registers are valid
        mutated = RegisterValidator.validate_and_fix_genetic_code(mutated)
        
        return mutated
    
    def evolve_population(self, population: List, fitnesses: np.ndarray, 
                        config: Dict) -> List[List]:
        """
        Create a new generation through selection, crossover and mutation.
        
        Args:
            population: Current population
            fitnesses: Fitness scores for population
            config: Evolution configuration
            
        Returns:
            New population
        """
        pop_size = config.get('population_size', 20)
        elite_count = config.get('elite_count', 1)
        tournament_size = config.get('tournament_size', 3)
        crossover_rate = config.get('crossover_rate', 0.7)
        mutation_rate = config.get('mutation_rate', 0.3)
        
        new_pop = []
        sorted_indices = sorted(range(len(population)), 
                              key=lambda i: fitnesses[i], 
                              reverse=True)
        
        # Add elite individuals
        for i in range(min(elite_count, len(sorted_indices))):
            elite = self.clone_function(population[sorted_indices[i]])
            # Validate elite individual
            elite = RegisterValidator.validate_and_fix_genetic_code(elite)
            new_pop.append(elite)
            
        # Fill the rest of the population
        while len(new_pop) < pop_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(population, fitnesses, tournament_size)
            parent2 = self._tournament_selection(population, fitnesses, tournament_size)
            
            # Crossover
            if random.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = self.clone_function(random.choice([parent1, parent2]))
                
            # Mutation
            if random.random() < mutation_rate:
                child = self.mutate(child, mutation_rate)
                
            # Validate child
            child = RegisterValidator.validate_and_fix_genetic_code(child)
            new_pop.append(child)
            
        return new_pop
    
    def _flexible_mutate_op_code(self, func: List, index: int) -> None:
        """Mutate operation code with possibility of changing categories."""
        if index >= len(func):
            return
            
        instr = func[index]
        current_op = instr[0]
        
        # Protect critical operations
        if current_op in (453, 454):  # RETURN, END
            return
        
        # Find the category of the current operation
        current_category = None
        for category, ops in self.op_categories.items():
            if current_op in ops:
                current_category = category
                break
        
        if current_category:
            # Choose a new operation from the same category
            category_ops = self.op_categories[current_category]
            new_op = random.choice(category_ops)
        else:
            # Fallback to a random arithmetic operation
            new_op = random.choice(self.op_categories["arithmetic"])
        
        # Get specifications for the new operation
        new_op_spec = self.op_registry.get(new_op)
        if not new_op_spec:
            return  # Skip if operation not in registry
        
        # Update the operation code
        func[index][0] = new_op
        
        # Adjust parameters to match the new operation's requirements
        self._adapt_parameters(func[index], new_op_spec)
    
    def _adapt_parameters(self, instr: List, op_spec: Dict) -> None:
        """Adapt instruction parameters to match the operation's requirements."""
        params_needed = op_spec["params"]
        param_types = op_spec["param_types"]
        
        # Current parameters (excluding op code)
        current_params = instr[1:]
        
        # Create a new parameter list with the right length
        new_params = []
        
        for i in range(params_needed):
            param_type = param_types[i]
            
            # Reuse existing parameter if possible and compatible
            if i < len(current_params):
                param = current_params[i]
            else:
                # Generate a new parameter based on type
                param = self._generate_param(param_type)
            
            # Ensure parameter value is appropriate for its type
            new_params.append(self._adapt_param_value(param, param_type))
        
        # Update the instruction with new parameters
        instr[1:] = new_params
    

    

    
    def _tournament_selection(self, population: List, fitnesses: np.ndarray, 
                            tournament_size: int = 3) -> List:
        """Perform tournament selection to pick a parent."""
        indices = list(range(len(population)))
        tournament_indices = random.sample(indices, min(tournament_size, len(indices)))
        
        best_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
                
        return population[best_idx]
    
    def _crossover(self, parent1: List, parent2: List) -> List:
        """Perform enhanced crossover between two parents."""
        p1 = self.clone_function(parent1)
        p2 = self.clone_function(parent2)
        
        if len(p1) < 2 or len(p2) < 2:
            return p1 if len(p1) >= len(p2) else p2
            
        # Choose crossover strategy
        strategy = random.randint(0, 2)
        
        if strategy == 0:
            # Single-point crossover
            point1 = random.randint(1, len(p1) - 1)
            point2 = random.randint(1, len(p2) - 1)
            child = p1[:point1] + p2[point2:]
        elif strategy == 1:
            # Two-point crossover
            if len(p1) >= 4 and len(p2) >= 4:
                points1 = sorted(random.sample(range(1, len(p1) - 1), 2))
                points2 = sorted(random.sample(range(1, len(p2) - 1), 2))
                child = p1[:points1[0]] + p2[points2[0]:points2[1]] + p1[points1[1]:]
            else:
                # Fall back to single-point crossover
                point1 = random.randint(1, len(p1) - 1)
                point2 = random.randint(1, len(p2) - 1)
                child = p1[:point1] + p2[point2:]
        else:
            # Uniform crossover
            child = []
            for i in range(max(len(p1), len(p2))):
                if i < len(p1) and i < len(p2):
                    child.append(p1[i] if random.random() < 0.5 else p2[i])
                elif i < len(p1):
                    child.append(p1[i])
                else:
                    child.append(p2[i])
        
        # Ensure valid child program
        if not child:
            return p1
            
        # Validate the child program registers
        child = RegisterValidator.validate_and_fix_genetic_code(child)
            
        return child
    
class SteadyStateGeneticAlgorithm(GeneticAlgorithmStrategy):
    """Steady-state genetic algorithm with winner-mutate/crossover strategy."""
    
    def __init__(self, config: Dict = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.op_registry = self._load_operation_registry()
        self.op_categories = self._load_operation_categories()
        
    
    def create_initial_population(self, seed: List, population_size: int = 20, 
                                 mutation_rate: float = 0.3) -> List[Tuple[List, int]]:
        """Create initial population with age tracking."""
        # Validate seed
        validated_seed = RegisterValidator.validate_and_fix_genetic_code(seed)
        
        # Initialize population as tuples of (genetic_code, age)
        population = [(self.clone_function(validated_seed), 0)]
        
        for _ in range(1, population_size):
            mutated = self.mutate(
                self.clone_function(validated_seed), 
                mutation_rate
            )
            population.append((mutated, 0))  # Age starts at 0
            
        return population
    
    def clone_function(self, func: List) -> List:
        """Make a deep copy of a genetic function."""
        return copy.deepcopy(func)
    
    def mutate(self, func: List, mutation_rate: float = 0.3) -> List:
        """
        Mutate a genetic function with various operations.
        
        Args:
            func: Function to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated function
        """
        mutated = self.clone_function(func)
        
        # Skip the first two instructions (LOAD operations) and the last instruction (RETURN)
        i = 2
        while i < len(mutated) - 1:
            if random.random() < mutation_rate:
                strategy = random.randint(0, 4)
                
                if strategy == 0:  # Modify op code
                    self._mutate_op_code(mutated, i)
                elif strategy == 1:  # Modify parameters
                    self._mutate_parameters(mutated, i)
                elif strategy == 2 and len(mutated) < 20:  # Insert instruction
                    self._insert_instruction(mutated, i)
                elif strategy == 3 and len(mutated) > 4:  # Delete instruction
                    if i < len(mutated):
                        self._delete_instruction(mutated, i)
                        continue  # Don't increment i since we removed an instruction
                elif strategy == 4:  # Swap instructions
                    self._swap_instructions(mutated, i)
            i += 1
        
        for idx in range(len(mutated)):
            self._fix_params(mutated[idx])


        # Ensure valid genetic code structure
        return RegisterValidator.validate_and_fix_genetic_code(mutated)
    
    def _tournament_select_two_best(self, population: List[Tuple[List, int]], 
                                   fitnesses: np.ndarray, 
                                   tournament_size: int = 3) -> Tuple[List, List]:
        """Select the best and second best individuals from a tournament."""
        indices = list(range(len(population)))
        tournament_indices = random.sample(indices, min(tournament_size, len(indices)))
        
        # Sort tournament participants by fitness (descending)
        sorted_tournament = sorted(tournament_indices, key=lambda i: fitnesses[i], reverse=True)
        
        if len(sorted_tournament) >= 2:
            best_idx = sorted_tournament[0]
            second_best_idx = sorted_tournament[1]
            return population[best_idx][0], population[second_best_idx][0]  # Return genetic codes only
        else:
            best_idx = sorted_tournament[0]
            # If only one individual in tournament, return it twice
            return population[best_idx][0], population[best_idx][0]
    
    def evolve_population_steady_state(self, population: List[Tuple[List, int]], 
                                     fitnesses: np.ndarray, 
                                     config: Dict) -> List[Tuple[List, int]]:
        """
        Evolve population using steady-state strategy.
        
        Args:
            population: Current population as list of (genetic_code, age) tuples
            fitnesses: Fitness scores for population
            config: Evolution configuration
            
        Returns:
            New population
        """
        tournament_size = config.get('tournament_size', 3)
        crossover_rate = config.get('crossover_rate', 0.7)
        mutation_rate = config.get('mutation_rate', 0.3)
        
        # Increment age of all individuals
        new_pop = [(genetic_code, age + 1) for genetic_code, age in population]
        
        # Find the oldest individual's index
        oldest_idx = max(range(len(new_pop)), key=lambda i: new_pop[i][1])
        
        # Tournament selection for winner and runner-up
        winner, runner_up = self._tournament_select_two_best(population, fitnesses, tournament_size)
        
        # Decide whether to mutate or crossover
        if random.random() < crossover_rate:
            # Crossover winner with runner-up
            child = self._crossover(winner, runner_up)
        else:
            # Mutate winner
            child = self.mutate(self.clone_function(winner), mutation_rate)
            
        # Validate child
        child = RegisterValidator.validate_and_fix_genetic_code(child)
        
        # Replace oldest individual
        new_pop[oldest_idx] = (child, 0)  # New individual has age 0
            
        return new_pop
    
    def evolve_population(self, population: List, fitnesses: np.ndarray, 
                         config: Dict) -> List:
        """
        Main evolution method - wrapper for steady-state evolution.
        
        Args:
            population: Current population (format depends on evolution stage)
            fitnesses: Fitness scores for population
            config: Evolution configuration
            
        Returns:
            New population
        """
        # If first generation, convert population to (genetic_code, age) format
        if not isinstance(population[0], tuple):
            population = [(genetic_code, 0) for genetic_code in population]
            
        # Number of iterations to perform
        iterations = config.get('steady_state_iterations', len(population))
        
        for _ in range(iterations):
            population = self.evolve_population_steady_state(population, fitnesses, config)
            
        # Return just the genetic codes for compatibility
        return [genetic_code for genetic_code, _ in population]
    
    def _crossover(self, parent1: List, parent2: List) -> List:
        """Perform crossover between two parents."""
        p1 = self.clone_function(parent1)
        p2 = self.clone_function(parent2)
        
        if len(p1) < 3 or len(p2) < 3:
            return p1 if len(p1) >= len(p2) else p2
            
        # Choose crossover strategy
        strategy = random.randint(0, 2)
        
        if strategy == 0:
            # Single-point crossover
            point1 = random.randint(2, len(p1) - 2)  # Preserve first two and last instruction
            point2 = random.randint(2, len(p2) - 2)
            child = p1[:point1] + p2[point2:]
        elif strategy == 1:
            # Two-point crossover
            if len(p1) >= 5 and len(p2) >= 5:
                points1 = sorted(random.sample(range(2, len(p1) - 1), 2))
                points2 = sorted(random.sample(range(2, len(p2) - 1), 2))
                child = p1[:points1[0]] + p2[points2[0]:points2[1]] + p1[points1[1]:]
            else:
                # Fall back to single-point
                point1 = random.randint(2, len(p1) - 2)
                point2 = random.randint(2, len(p2) - 2)
                child = p1[:point1] + p2[point2:]
        else:
            # Uniform crossover
            child = [p1[0], p1[1]]  # Preserve first two LOAD instructions
            for i in range(2, min(len(p1), len(p2)) - 1):
                child.append(p1[i] if random.random() < 0.5 else p2[i])
            child.append([453, 9])  # Ensure RETURN instruction
        
        # Validate the child program
        return RegisterValidator.validate_and_fix_genetic_code(child)
        
    # Helper methods for mutation (simplified versions)
    def _mutate_op_code(self, func: List, index: int) -> None:
        """Mutate operation code."""
        if index >= len(func) or func[index][0] in (453, 454):  # Don't mutate RETURN or END
            return
            
        # Find current operation category
        current_op = func[index][0]
        current_category = None
        for category, ops in self.op_categories.items():
            if current_op in ops:
                current_category = category
                break
                
        # Choose a new operation from the same category
        if current_category:
            new_op = random.choice(self.op_categories[current_category])
            func[index][0] = new_op
    
    def _mutate_parameters(self, func: List, index: int) -> None:
        """Mutate instruction parameters."""
        if index >= len(func) or len(func[index]) <= 1:
            return
            
        # Pick a random parameter to mutate (excluding op code)
        param_idx = random.randint(1, len(func[index]) - 1)
        
        # For simplicity, just use a reasonable value range
        if param_idx == len(func[index]) - 1:  # Last param often destination register
            func[index][param_idx] = random.randint(3, 15)  # General register
        else:
            func[index][param_idx] = random.randint(1, 5)  # Source register or small value
    
    def _insert_instruction(self, func: List, index: int) -> None:
        """Insert a new instruction."""
        # Pick a random operation category
        category = random.choice(list(self.op_categories.keys()))
        if category == "control":  # Avoid control operations
            category = "arithmetic"
            
        # Get a random operation from this category
        op_code = random.choice(self.op_categories[category])
        
        # Create a simple instruction with basic parameters
        if op_code <= 12:  # Basic arithmetic with 3 params
            new_instr = [op_code, random.randint(1, 5), random.randint(1, 5), random.randint(3, 15)]
        elif op_code in (100, 101):  # LOAD operations
            new_instr = [op_code, random.randint(3, 15), random.randint(0, 1)]
        else:  # Default format
            new_instr = [op_code, random.randint(1, 5), random.randint(3, 15)]
            
        func.insert(index, new_instr)
    
    def _delete_instruction(self, func: List, index: int) -> None:
        """Delete an instruction."""
        # Don't delete if too few instructions or if it's a critical instruction
        if len(func) <= 4 or index < 2 or index >= len(func) - 1:
            return
        func.pop(index)
    
    def _swap_instructions(self, func: List, index: int) -> None:
        """Swap instruction with another non-critical one."""
        if len(func) <= 4 or index < 2 or index >= len(func) - 1:
            return
            
        # Find another valid index to swap with
        valid_indices = [i for i in range(2, len(func) - 1) if i != index]
        if not valid_indices:
            return
            
        other_idx = random.choice(valid_indices)
        func[index], func[other_idx] = func[other_idx], func[index]
