"""Evolution Engine for Genetic Function Optimization in PyTorch."""

import torch
import numpy as np
import logging
import traceback
import json
from typing import Dict, List, Optional, Any, Type
import torch
import numpy as np
import logging
import traceback
import time       # <- NEW

from .genetic.strategies import GeneticAlgorithmStrategy, StandardGeneticAlgorithm, SteadyStateGeneticAlgorithm
from .genetic.interpreter import GeneticInterpreter # May become part of component strategy
from .genetic.function_converter import FunctionConverter # May become part of component strategy
from .components import AIComponentStrategy, ComponentStrategyFactory

# Configure logger for this module
logger = logging.getLogger(__name__)

def log_genetic_code_stats(genetic_code: List) -> None:
    """Log statistics about a genetic code."""
    try:
        if not isinstance(genetic_code, list):
            logger.warning(f"Invalid genetic code type: {type(genetic_code)}")
            return
            
        op_codes = {}
        for instruction in genetic_code:
            if not isinstance(instruction, list) or len(instruction) == 0:
                continue
                
            op_code = instruction[0]
            op_codes[op_code] = op_codes.get(op_code, 0) + 1
            
        logger.debug(f"Genetic code length: {len(genetic_code)}")
        logger.debug(f"Operation code distribution: {json.dumps(op_codes)}")
        
        # Check for potentially problematic operation codes
        for op_code in op_codes:
            if 200 <= op_code < 250 and op_code > 203 and op_code != 205:
                logger.warning(f"Genetic code contains unsupported reduction operation: {op_code}")
    except Exception as e:
        logger.warning(f"Error logging genetic code stats: {e}")

class EvolutionEngine:
    """Handles the evolutionary process for various AI components."""

    def __init__(self,
                 component_type: str,
                 config: Optional[Dict] = None,
                 genetic_strategy: Optional[GeneticAlgorithmStrategy] = None,
                 component_strategy_class: Optional[Type[AIComponentStrategy]] = None):
        """
        Initialize the evolution engine for a specific AI component.

        Args:
            component_type: The type of AI component to evolve (e.g., 'loss', 'optimizer').
                            Used if component_strategy_class is not provided.
            config: Dictionary containing configuration settings for the engine and strategies.
            genetic_strategy: Strategy for genetic algorithm operations (mutation, crossover, etc.).
            component_strategy_class: Explicit class for the component strategy. If provided,
                                      component_type is ignored for factory lookup.
        """
        # Default configuration
        default_config = {
            'population_size': 200,
            'generations': 10,
            'mutation_rate': 0.2,
            'tournament_size': 10,
            'batch_size': 32, # May be used by component strategy's evaluate method
            'elite_count': 2,
            'crossover_rate': 0.7,
            'evaluation_rounds': 1, # May be used by component strategy's evaluate method
            'debug_logging': True,
            'log_interval': 100,
            'debug_level': 2,
            'component_config': {}, # Specific config for the component strategy
            'stop_flag': None  # Add stop_flag to default config
        }
        self.config = {**default_config, **(config or {})}
        
        # Extract stop_flag from config
        self.stop_flag = self.config.get('stop_flag')

        # --- Strategy Initialization ---
        # Genetic Strategy (handles mutation, crossover, population management)
        self.genetic_strategy = genetic_strategy or SteadyStateGeneticAlgorithm(self.config)

        # Component Strategy (handles creation, evaluation, interpretation for a specific AI component)
        if component_strategy_class:
            self.component_strategy = component_strategy_class(self.config)
        else:
            if not component_type:
                raise ValueError("Either component_type or component_strategy_class must be provided.")
            self.component_strategy = ComponentStrategyFactory.create_strategy(
                component_type, self.config
            )
        self.component_type = component_type or self.component_strategy.__class__.__name__.replace("Strategy", "").lower()
        # --- End Strategy Initialization ---

        self.is_running = False
        self.current_task_id = None # Store task ID instead of the whole task dict
        logger.info(f"EvolutionEngine initialized for component type '{self.component_type}'")
        logger.debug(f"Engine Config: {json.dumps({k: v for k, v in self.config.items() if k != 'stop_flag'}, indent=2)}")
        
    def should_stop(self):
        """
        Check if evolution should stop.
        
        Returns:
            True if evolution should stop, False otherwise
        """
        # Check shared stop flag directly
        if self.stop_flag and self.stop_flag.is_stopped():
            logger.info("Evolution stopping: stop flag triggered")
            return True
        return False

    def evolve_component(self, task: Dict) -> Optional[Dict]:
        """
        Evolve a genetic representation for a specific AI component.

        Args:
            task: Dictionary containing task details:
                - functions: List of parent functions/components [{'id': str, 'function': List}]
                - component_type: Type of component being evolved (redundant if engine is type-specific)
                - batch_id: Identifier for this evolution task
                - dataset: Pre-prepared dataset dictionary (Optional, depends on component strategy)
                - model: Pre-initialized model if needed by evaluation (Optional)

        Returns:
            Dictionary containing the evolved component details, or None if error or stopped.
                - evolved_representation: Python function string or other representation
                - parent_functions: List of parent function IDs
                - metadata: Evolution metadata (fitness, stats, etc.)
                - genetic_code: The raw genetic code of the best individual
                - task_interpretation: Component-specific interpretation
        """
        # Check for stop before starting
        if self.should_stop():
            logger.info("Evolution cancelled before starting: stop flag triggered")
            return None
            
        if self.is_running:
            logger.warning(f"Evolution already running (Task ID: {self.current_task_id}). Cannot start new task {task.get('batch_id')}")
            return None

        task_id = task.get('batch_id', 'unknown_task')
        try:
            self.is_running = True
            self.current_task_id = task_id

            # Extract task details
            logger.info(f"Starting evolution task: {task_id}")
            logger.info(f"Component type from task: {task.get('component_type', 'N/A')} (Engine type: {self.component_type})")
            logger.debug(f"Parent functions received: {len(task.get('functions', []))} functions")
            if not task.get('functions'):
                 raise ValueError("Task must include 'functions' list with at least one parent.")
                 
            # Check for stop after validation
            if self.should_stop():
                logger.info("Evolution cancelled after validation: stop flag triggered")
                self.is_running = False
                self.current_task_id = None
                return None
                 
            # Convert parent functions to genetic code using component strategy
            parent_info = task['functions'][0]
            parent_id = parent_info.get('id', 'unknown_parent')
            logger.debug(f"Processing parent function {parent_id}")
            parent_genetic_code = parent_info.get('function')
            
            # Format and log the parent genetic code
            if parent_genetic_code:
                if isinstance(parent_genetic_code, list):
                    logger.debug(f"Parent genetic code length: {len(parent_genetic_code)}")
                    logger.debug(f"Parent genetic code structure:\n" + 
                                 "\n".join([f"{i}: {instr}" for i, instr in enumerate(parent_genetic_code)]))
                else:
                    logger.warning(f"Parent genetic code has invalid type: {type(parent_genetic_code)}")
            else:
                logger.warning("No parent genetic code provided")

            # Get initial genetic code
            initial_genetic_code = parent_genetic_code if isinstance(parent_genetic_code, list) else None

            # If no genetic code provided, use component strategy's initial solution
            if not initial_genetic_code:
                logger.info("No initial genetic code provided, creating one using component strategy.")
                initial_genetic_code = self.component_strategy.create_initial_solution()
                if not initial_genetic_code:
                     raise ValueError("Component strategy failed to create an initial solution.")

            logger.info(f"Initial genetic code length: {len(initial_genetic_code)}")

            # Log genetic code details
            if self.config['debug_level'] >= 1:
                log_genetic_code_stats(initial_genetic_code)
                
            # Check for stop before population creation
            if self.should_stop():
                logger.info("Evolution cancelled before population creation: stop flag triggered")
                self.is_running = False
                self.current_task_id = None
                return None

            # Create initial population using genetic strategy
            logger.info(f"Creating initial population of size {self.config['population_size']}")
            population = self.genetic_strategy.create_initial_population(
                initial_genetic_code,
                population_size=self.config['population_size'],
                mutation_rate=self.config['mutation_rate'] # Pass relevant config
            )

            # Log population details
            if self.config['debug_level'] >= 2:
                logger.debug(f"Initial population created with {len(population)} individuals")
                for i, individual in enumerate(population[:3]): # Log first 3 individuals
                    logger.debug(f"Individual {i} length: {len(individual)}")
                    log_genetic_code_stats(individual)

            # Main evolutionary loop
            best_individual = None
            best_fitness = -float('inf')
            generation_stats = []

            logger.info(f"Starting evolution for {self.config['generations']} generations")
            for generation in range(self.config['generations']):
                # Check for stop at the beginning of each generation
                if self.should_stop():
                    logger.info(f"Evolution stopped at generation {generation}/{self.config['generations']}")
                    # Use the best individual found so far if evolution was partially completed
                    if best_individual is not None:
                        logger.info(f"Using best individual found so far (fitness: {best_fitness})")
                        break
                    else:
                        self.is_running = False
                        self.current_task_id = None
                        return None
                
                # Evaluate population
                if generation % self.config.get('log_interval', 100) == 0: # Use config value
                    logger.info(f"Generation {generation}/{self.config['generations']}")

                # Log before evaluation
                if self.config['debug_level'] >= 2 and generation % self.config.get('log_interval', 100) == 0:
                    logger.debug(f"Generation {generation}: Evaluating population of {len(population)} individuals")

                # Pass dataset and potentially model to evaluation, plus stop_flag through self
                current_fitnesses = self._evaluate_population(population)
                
                # Check if evaluation was interrupted by stop flag
                if self.should_stop() and len([f for f in current_fitnesses if f > -float('inf')]) == 0:
                    logger.info(f"Population evaluation interrupted at generation {generation}")
                    # Use the best individual found so far if evolution was partially completed
                    if best_individual is not None:
                        logger.info(f"Using best individual found so far (fitness: {best_fitness})")
                        break
                    else:
                        self.is_running = False
                        self.current_task_id = None
                        return None

                # Find best individual in this generation
                if len(current_fitnesses) == 0:
                    logger.warning(f"Generation {generation}: Evaluation returned no fitness scores. Skipping generation.")
                    
                    # Check for stop after failed evaluation
                    if self.should_stop():
                        if best_individual is not None:
                            logger.info(f"Using best individual found so far (fitness: {best_fitness})")
                            break
                        else:
                            self.is_running = False
                            self.current_task_id = None
                            return None
                            
                    continue # Or handle error appropriately
                    
                best_idx = np.argmax(current_fitnesses)
                generation_best_fitness = current_fitnesses[best_idx]

                # Log stats at regular intervals or if there's an improvement
                should_log = (
                    self.config['debug_logging'] and
                    (generation % self.config.get('log_interval', 100) == 0 or
                     generation_best_fitness > best_fitness)
                )

                # Update overall best if improved
                if generation_best_fitness > best_fitness:
                    best_individual = self.genetic_strategy.clone_function(population[best_idx])
                    best_fitness = generation_best_fitness
                    logger.info(f"Generation {generation}: New best fitness: {best_fitness:.6f}")

                    # Log best individual details
                    if self.config['debug_level'] >= 1:
                        logger.debug(f"New best individual length: {len(best_individual)}")
                        log_genetic_code_stats(best_individual)

                if should_log:
                    # Calculate and log statistics for this generation
                    valid_fitnesses = current_fitnesses[~np.isnan(current_fitnesses)] # Exclude NaNs
                    if len(valid_fitnesses) > 0:
                        avg_fitness = np.mean(valid_fitnesses)
                        min_fitness = np.min(valid_fitnesses)
                        median_fitness = np.median(valid_fitnesses)
                        logger.info(f"Generation {generation} stats: "
                                    f"Best={generation_best_fitness:.6f}, "
                                    f"Avg={avg_fitness:.6f}, "
                                    f"Min={min_fitness:.6f}, "
                                    f"Median={median_fitness:.6f}")
                        # Store stats for later analysis
                        generation_stats.append({
                            'generation': generation,
                            'best': generation_best_fitness,
                            'avg': avg_fitness,
                            'min': min_fitness,
                            'median': median_fitness
                        })
                    else:
                         logger.warning(f"Generation {generation}: No valid fitness scores to calculate stats.")

                # Check for stop before creating next generation
                if self.should_stop():
                    logger.info(f"Evolution stopped after generation {generation} evaluation")
                    # Use the best individual found so far
                    if best_individual is not None:
                        logger.info(f"Using best individual found so far (fitness: {best_fitness})")
                        break
                    else:
                        self.is_running = False
                        self.current_task_id = None
                        return None

                # Create next generation
                if generation < self.config['generations'] - 1:
                    # Pass evolution parameters from config
                    evo_options = {
                        'population_size': self.config['population_size'],
                        'elite_count': self.config['elite_count'],
                        'tournament_size': self.config['tournament_size'],
                        'crossover_rate': self.config['crossover_rate'],
                        'mutation_rate': self.config['mutation_rate']
                    }
                    new_population = self.genetic_strategy.evolve_population(
                        population,
                        current_fitnesses, # Use fitnesses from this generation
                        evo_options
                    )
                    population = new_population

            # End of evolution loop
            
            # Check if we have a valid result
            if best_individual is None:
                logger.error("Evolution finished without finding a best individual.")
                self.is_running = False
                self.current_task_id = None
                return None

            # Convert best genetic code to a usable representation (handled by interpret_result)
            logger.info(f"Evolution complete. Best fitness: {best_fitness}")
            logger.info(f"Best genetic code length: {len(best_individual)}")
            
            # Check for stop before final interpretation
            if self.should_stop():
                logger.info("Evolution stopped before final interpretation")
                self.is_running = False
                self.current_task_id = None
                # We'll still return a result if we have a best individual
                if best_individual is None:
                    return None

            # Create result object for interpretation
            evolution_result_data = {
                'fitness': best_fitness,
                'genetic_code': best_individual,
                'generations': self.config['generations'],
                'population_size': self.config['population_size'],
                'parent_id': parent_id,
                'genetic_code_length': len(best_individual),
                'total_evaluations': self.config['population_size'] * self.config['generations'],
                'generation_stats': generation_stats, # Include detailed stats
                'completed': not self.should_stop() # Flag to indicate if evolution completed normally
            }

            # Interpret result using component strategy
            task_interpretation = self.component_strategy.interpret_result(evolution_result_data)
            # The interpretation might include the 'python_function' or other representations
            evolved_representation = task_interpretation.get('evolved_representation', None) # e.g., python code string

            logger.info(f"Evolution task {task_id} finished. Interpretation: {json.dumps(task_interpretation, indent=2)}")

            self.is_running = False
            self.current_task_id = None

            # Return the final results
            return {
                'evolved_representation': evolved_representation, # Could be code, config dict, etc.
                'parent_functions': [{'id': parent_id}], # Keep track of lineage
                'metadata': {**evolution_result_data, 'task_interpretation': task_interpretation}, # Combine all metadata
                'genetic_code': best_individual, # Raw genetic code
                'task_interpretation': task_interpretation # Component-specific results
            }

        except Exception as e:
            error_details = traceback.format_exc()
            logger.exception(f"Error during evolution task {task_id}: {str(e)}")
            logger.debug(f"Evolution error details: {error_details}")
            self.is_running = False
            self.current_task_id = None
            return None

    def _evaluate_population(self, population: List) -> np.ndarray:
        """Evaluate all individuals in the population using the component strategy."""
        start_ts   = time.perf_counter()        
        eval_count = 0                          
        fitnesses = np.full(len(population), -np.inf) # Initialize with worst possible fitness

        for i, individual in enumerate(population):
            # Check for stop signal periodically (every individual)
            if i % 1 == 0 and self.should_stop():  # Check every individual
                logger.info(f"Population evaluation interrupted at individual {i}/{len(population)}")
                return fitnesses  # Return partial results
                
            try:
                # Log individual details before evaluation
                if self.config['debug_level'] >= 2 and i < 5: # Log first 5 individuals
                    logger.debug(f"Evaluating individual {i}, code length: {len(individual)}")
                    # Optional: Add more detailed logging like log_genetic_code_stats(individual)

                # Evaluate the individual using the component strategy
                # Pass stop_flag to evaluation
                fitness = self.component_strategy.evaluate(individual, stop_flag=self.stop_flag)
                
                # Check if evaluation was interrupted
                if fitness is None and self.should_stop():
                    logger.info(f"Evaluation of individual {i} was interrupted")
                    return fitnesses  # Return partial results

                # Handle potential NaN or invalid fitness values
                if fitness is None or np.isnan(fitness):
                    logger.warning(f"Individual {i} evaluation returned invalid fitness (None or NaN). Assigning minimal fitness.")
                    fitnesses[i] = -np.inf # Assign a very low fitness
                else:
                    fitnesses[i] = float(fitness)

                eval_count += 1

                # Log successful evaluation
                if self.config['debug_level'] >= 2 and i < 5: # Log first 5 individuals
                    logger.debug(f"Individual {i} evaluated successfully, fitness: {fitnesses[i]:.6f}")
                
            except Exception as e:
                # Log error without stopping the entire evaluation
                error_details = traceback.format_exc()
                logger.error(f"Error evaluating individual {i}: {str(e)}")
                if self.config['debug_level'] >= 1:
                    logger.debug(f"Evaluation error details for individual {i}: {error_details}")
                    # Log genetic code that caused the error
                    try:
                        logger.debug(f"Problematic genetic code (Ind {i}): {json.dumps(individual)}")
                    except Exception as json_err:
                        logger.debug(f"Could not serialize problematic genetic code: {json_err}")

                fitnesses[i] = -np.inf # Assign worst fitness on error
                
            # Check for stop after each evaluation
            if self.should_stop():
                logger.info(f"Population evaluation interrupted after individual {i}/{len(population)}")
                return fitnesses  # Return partial results

        # Log overall evaluation results
        if self.config['debug_level'] >= 1:
            valid_fitnesses = fitnesses[np.isfinite(fitnesses)]
            successful_count = len(valid_fitnesses)
            avg_fitness = np.mean(valid_fitnesses) if successful_count > 0 else 0
            logger.debug(f"Population evaluation complete: {successful_count}/{len(population)} successful evaluations. "
                        f"Avg valid fitness: {avg_fitness:.6f}")
            elapsed = max(time.perf_counter() - start_ts, 1e-9)   # Avoid /0
            eps     = eval_count / elapsed
            logger.info(f"Throughput: {eval_count} evaluations in {elapsed:.3f} s "
                        f"→ {eps:.2f} eval/s")

        return fitnesses
