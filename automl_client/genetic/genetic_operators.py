"""Genetic Operators for Neural Genetic Language in PyTorch (legacy wrapper)."""

from .strategies import StandardGeneticAlgorithm, SteadyStateGeneticAlgorithm
from .operators import generate_random_program, log_genetic_code_stats, weighted_choice

# Maintain backward compatibility by creating a default instance
_default_strategy = SteadyStateGeneticAlgorithm()

class GeneticOperators:
    """Legacy wrapper for genetic operators (use StandardGeneticAlgorithm instead)."""
    """Genetic operators with more flexible mutation strategies."""
    
    OP_REGISTRY = _default_strategy.op_registry
    OP_CATEGORIES = _default_strategy.op_categories
    
    @staticmethod
    def create_initial_population(seed, population_size=20, mutation_rate=0.3, 
                                cross_category_rate=1.0):
        """Create initial population (legacy wrapper)."""
        from .genetic_library import GeneticLibrary
        seed_function = GeneticLibrary.get_function(seed) if isinstance(seed, str) else seed
        return _default_strategy.create_initial_population(
            seed_function, population_size, mutation_rate
        )
    
    @staticmethod
    def create_random_population(population_size=20, complexity=5, favor_categories=None):
        """Create random population (legacy wrapper)."""
        population = []
        for _ in range(population_size):
            population.append(generate_random_program(complexity, favor_categories))
        return population
    
    @staticmethod
    def clone_function(func):
        """Clone function (legacy wrapper)."""
        return _default_strategy.clone_function(func)

    @staticmethod
    def mutate(func, mutation_rate=0.3, cross_category_rate=0.3):
        """Mutate function (legacy wrapper)."""
        return _default_strategy.mutate(func, mutation_rate)
    
    
    @staticmethod
    def generate_random_program(complexity=5, favor_categories=None):
        """
        Generate a random genetic program with optional category preference.
        
        Args:
            complexity: Approximate program length
            favor_categories: Optional list of categories to favor (e.g., ["arithmetic", "nn"])
            
        Returns:
            Random genetic program
        """
        program = []
        actual_complexity = max(3, complexity + random.randint(-1, 1))
        
        # Always start with loading inputs
        program.append([100, 1, 0])  # LOAD R1, S0 (y_true)
        program.append([100, 2, 1])  # LOAD R2, S1 (y_pred)
        
        # Set up category weights for selection
        categories = list(GeneticOperators.OP_CATEGORIES.keys())
        weights = [1.0] * len(categories)
        
        # Increase weights for favored categories
        if favor_categories:
            for i, category in enumerate(categories):
                if category in favor_categories:
                    weights[i] = 3.0  # Triple the probability
        
        # Generate the middle of the program
        for i in range(actual_complexity):
            # Select a category based on weights
            category_idx = GeneticOperators._weighted_choice(weights)
            category = categories[category_idx]
            
            # Avoid control flow operations in the middle
            if category == "control":
                category = "arithmetic"
                
            # Choose an operation from this category
            op_code = random.choice(GeneticOperators.OP_CATEGORIES[category])
            op_spec = GeneticOperators.OP_REGISTRY.get(op_code)
            
            if not op_spec:
                continue  # Skip if operation not in registry
                
            # Create new instruction
            new_instr = [op_code]
            
            # Register for storing result (typically increases with program complexity)
            dest_reg = 3 + i
            
            # Add appropriate parameters based on operation type
            for j, param_type in enumerate(op_spec["param_types"]):
                if j == len(op_spec["param_types"]) - 1 and param_type == "reg":
                    # Last parameter is typically the destination register
                    new_instr.append(dest_reg)
                else:
                    # For source registers, prefer using existing registers
                    if param_type == "reg" and i > 0:
                        # Choose from registers that have been assigned values
                        new_instr.append(random.randint(1, 2 + i))
                    else:
                        new_instr.append(GeneticOperators._generate_param(param_type))
            
            program.append(new_instr)
        
        # Add a reduction at the end to get a scalar
        program.append([201, 3 + actual_complexity - 1, -1, 9])  # REDUCE_MEAN to output register
        
        # Always end with RETURN
        program.append([453, 9])  # RETURN output register
        
        return program
    
    @staticmethod
    def _weighted_choice(weights):
        """
        Select an index based on weights.
        
        Args:
            weights: List of weights
            
        Returns:
            Selected index
        """
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return i
                
        return len(weights) - 1  # Fallback
    
    @staticmethod
    def evolve_population(population, fitnesses, options=None):
        """Evolve population (legacy wrapper)."""
        return _default_strategy.evolve_population(population, fitnesses, options or {})
