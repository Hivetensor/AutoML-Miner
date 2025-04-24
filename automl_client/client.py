"""Client API for connecting to the Neural Component Pool."""

import json
import uuid
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
import torch
import abc
import logging
from automl_client.data.dataset_factory import DatasetFactory
from automl_client.components import ComponentStrategyFactory
from automl_client.wallet import Wallet 

logger = logging.getLogger(__name__)

from .genetic.serialize import (
    serialize_genetic_code,
    deserialize_genetic_code,
    serialize_population,
    deserialize_population,
    serialize_evolution_result,
    deserialize_evolution_result
)

from .evolution_engine import EvolutionEngine
from .evaluation_strategy import DefaultPytorchEvaluation, OptimizerEvaluation

class PoolClient(abc.ABC):
    """Client for interacting with the Neural Component Pool server."""
    
    def __init__(self, 
                 public_address: str,
                 base_url: str = "http://localhost:8000"):
        """
        Initialize the client with authentication credentials.
        
        Args:
            public_address: Public address for authentication
            base_url: Base URL for the API server
        """
        self.public_address = public_address
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/api/v1"  # Add API prefix
        self.session = None
        self._stop_mining = False  # Initialize the stop flag
        
    def stop_mining(self):
        """Flag to stop ongoing mining operations."""
        logger.info("Stop mining flag set")
        self._stop_mining = True
        
    def should_stop(self, stop_flag=None):
        """
        Check if mining should stop.
        
        Args:
            stop_flag: Optional StopFlag object
            
        Returns:
            True if mining should stop, False otherwise
        """
        # Check internal flag first
        if self._stop_mining:
            return True
        
        # Check shared stop flag if provided
        if stop_flag and stop_flag.is_stopped():
            return True
                
        return False

    def __enter__(self):
        """Set up client session context manager."""
        self.session = requests.Session()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up client session."""
        if self.session:
            self.session.close()
            self.session = None
    
    @abc.abstractmethod
    def _create_signature_data(self) -> Dict:
        """
        Create authentication data for API requests.
        
        Returns:
            Dictionary with public_address, signature, and message
        """
        pass
    
    def _ensure_session(self):
        """Ensure an HTTP session exists."""
        if not self.session:
            self.session = requests.Session()

    def register(self) -> Dict:
        """
        Register miner with the pool.
        
        Returns:
            Registration response
        """
        self._ensure_session()
        
        auth_data = self._create_signature_data()
        response = self.session.post(
            f"{self.base_url}{self.api_prefix}/miners/register",
            json=auth_data
        )
        response.raise_for_status()
        return response.json()
    
    def get_miner_info(self) -> Dict:
        """
        Get miner information.
        
        Returns:
            Miner information
        """
        self._ensure_session()
        
        response = self.session.get(
            f"{self.base_url}{self.api_prefix}/miners/{self.public_address}"
        )
        response.raise_for_status()
        return response.json()
    
    def get_balance(self) -> Dict:
        """
        Get Alpha token balance.
        
        Returns:
            Balance information
        """
        self._ensure_session()
        
        response = self.session.get(
            f"{self.base_url}{self.api_prefix}/miners/{self.public_address}/balance"
        )
        response.raise_for_status()
        return response.json()
    
    def request_task(self, task_type: str, max_retries: int = 3, stop_flag=None) -> Dict:
        """
        Request a task from the pool with automatic error handling.
        If "Miner already has active task" error occurs, automatically resets and retries.
        
        Args:
            task_type: Type of task to request ("evolve" or "evaluate")
            max_retries: Maximum number of retry attempts
            stop_flag: Optional StopFlag object to check for stopping
                
        Returns:
            Task batch data or None if stopped
        """
        # Check for stop signal before starting
        if self.should_stop(stop_flag):
            logger.info("Task request cancelled: stop flag triggered")
            return None
            
        self._ensure_session()
        retry_count = 0
        
        while retry_count < max_retries:
            # Check stop flag in the retry loop too
            if self.should_stop(stop_flag):
                logger.info("Task request cancelled during retry: stop flag triggered")
                return None
                
            try:
                auth_data = self._create_signature_data()
                data = {
                    **auth_data,
                    "task_type": task_type
                }
                
                if retry_count == 0:
                    logger.info(f"Requesting {task_type} task...")
                else:
                    logger.info(f"Retry #{retry_count}: Requesting {task_type} task...")
                
                response = self.session.post(
                    f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/request",
                    json=data
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                # Check if it's the "already has active task" error
                if e.response and e.response.status_code == 400:
                    try:
                        error_detail = e.response.json().get("detail", "")
                        if "already has active task" in error_detail:
                            logger.warning(f"Error: {error_detail}")
                            
                            # Try to reset active tasks directly via SQL
                            logger.info("Attempting to reset active tasks via direct database update...")
                            try:
                                # Try to reset via API first
                                self._direct_reset_active_tasks()
                                logger.info("Reset successful, retrying request...")
                                retry_count += 1
                                time.sleep(1)  # Brief pause before retry
                                continue
                            except Exception as reset_error:
                                logger.error(f"Reset failed: {reset_error}")
                    except Exception:
                        pass
                
                # If we get here, it's not a task error or reset failed
                logger.error(f"Request failed: {e}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error in request_task: {e}")
                raise
                
            retry_count += 1
        
        raise RuntimeError(f"Failed to request task after {max_retries} attempts")
    
    def _direct_reset_active_tasks(self):
        """
        Reset active tasks using direct API call.
        Internal method used by request_task for error recovery.
        """
        auth_data = self._create_signature_data()
        
        # First try the API endpoint if it exists
        try:
            response = self.session.post(
                f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/reset_active",
                json=auth_data,
                timeout=5  # Short timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Reset via API: {result}")
            return result
        except Exception as e:
            logger.warning(f"API reset failed (expected if endpoint not implemented): {e}")
            
            # If API endpoint doesn't exist, try a direct task creation
            # This will likely fail but might trigger the server to clean up stale tasks
            try:
                dummy_data = {
                    **auth_data,
                    "force_reset": True  # Custom flag that might be recognized by server
                }
                
                response = self.session.post(
                    f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/request",
                    json=dummy_data,
                    timeout=5  # Short timeout
                )
                # We don't expect this to succeed, but it might help
            except Exception:
                pass
                
            # Wait a moment for any server cleanup to complete
            time.sleep(2)
            
            return {"status": "attempted_reset", "message": "Attempted alternate reset methods"}

    def submit_evolution(self, batch_id: Union[str, uuid.UUID], 
                       evolved_function: str,
                       parent_functions: List[Dict],
                       metadata: Optional[Dict] = None) -> Dict:
        """
        Submit evolved function.
        
        Args:
            batch_id: Batch ID from task request
            evolved_function: Evolved function code
            parent_functions: List of parent function references
            metadata: Optional metadata about the evolution
            
        Returns:
            Submission response
        """
        self._ensure_session()
        
        auth_data = self._create_signature_data()
        data = {
            **auth_data,
            "evolved_function": evolved_function,
            "parent_functions": parent_functions,
            "metadata": metadata or {}
        }
        
        response = self.session.post(
            f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/{batch_id}/submit_evolution",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def submit_evaluation(self, batch_id: Union[str, uuid.UUID],
                        evaluations: List[Dict]) -> Dict:
        """
        Submit evaluations.
        
        Args:
            batch_id: Batch ID from task request
            evaluations: List of function evaluations
            
        Returns:
            Submission response
        """
        self._ensure_session()
        
        auth_data = self._create_signature_data()
        data = {
            **auth_data,
            "batch_id": str(batch_id),
            "evaluations": evaluations
        }
        
        response = self.session.post(
            f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/{batch_id}/submit_evaluation",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def process_evolution_task(self, task_data: Dict, stop_flag=None) -> Dict:
        """Process an evolution task."""
        # Check for stop before processing
        if self.should_stop(stop_flag):
            logger.info("Evolution task processing cancelled: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered"}
            
        # Extract task information
        batch_id = task_data["batch_id"]
        component_type = task_data["component_type"]
        # Map legacy or pool-specific types to internal types
        if component_type == "loss_function":
            component_type = "loss"
        elif component_type == "activation_function":
            component_type = "activation"

        functions = task_data["functions"]
        
        # Deserialize genetic code in functions if needed
        for i, func in enumerate(functions):
            if isinstance(func["function"], str):
                try:
                    functions[i]["function"] = deserialize_genetic_code(func["function"])
                except ValueError:
                    # Not a serialized genetic code, leave as is
                    pass
        
        # Evaluate all parent functions to find the best parent fitness
        component_strategy = ComponentStrategyFactory.create_strategy(component_type)
        best_parent_fitness = -float('inf')
        best_parent_id = None
        
        logger.info(f"Evaluating {len(functions)} parent functions")
        for parent in functions:
            # Check for stop signal during parent evaluation
            if self.should_stop(stop_flag):
                logger.info("Parent evaluation cancelled: stop flag triggered")
                return {"status": "cancelled", "reason": "Stop flag triggered during parent evaluation"}
                
            parent_id = parent.get("id", "unknown_parent")
            parent_genetic_code = parent["function"]
            
            try:
                # Pass stop_flag to evaluation
                parent_fitness = component_strategy.evaluate(parent_genetic_code, stop_flag=stop_flag)
                logger.info(f"Parent function (ID: {parent_id}) fitness: {parent_fitness}")
                
                if parent_fitness > best_parent_fitness:
                    best_parent_fitness = parent_fitness
                    best_parent_id = parent_id
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate parent function {parent_id}: {e}")
                # Continue with other parents if one fails
        
        logger.info(f"Best parent fitness: {best_parent_fitness} (ID: {best_parent_id})")
        
        # Create a task object for the evolution engine
        task = {
            "functions": functions,
            "component_type": component_type,
            "batch_id": batch_id
        }

        # Create config for EvolutionEngine that includes the stop flag
        engine_config = None
        if stop_flag:
            engine_config = {'stop_flag': stop_flag}
            
        # Instantiate Engine Dynamically based on component_type from the task
        try:
            # Pass the config with stop_flag if available
            evolution_engine = EvolutionEngine(component_type=component_type, config=engine_config)
            logger.info(f"Created EvolutionEngine for component type: {component_type}")
        except ValueError as e:
            logger.error(f"Failed to create EvolutionEngine for unknown type '{component_type}': {e}")
            raise RuntimeError(f"Unsupported component type received from pool: {component_type}") from e

        # Check for stop before starting evolution
        if self.should_stop(stop_flag):
            logger.info("Evolution cancelled before starting: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered before evolution"}
            
        # Evolve the component using the dynamically created engine
        result = evolution_engine.evolve_component(task)
        
        # Check if evolution was cancelled
        if result is None and self.should_stop(stop_flag):
            logger.info("Evolution was cancelled by stop flag")
            return {"status": "cancelled", "reason": "Evolution cancelled by stop flag"}
            
        logger.info(f"Evolution result structure: {type(result)}")
        logger.debug(f"Evolution result content: {result}")

        if not result:
            raise RuntimeError("Evolution failed")
        
        # Check for problematic values
        def check_problematic_values(obj, path=""):
            """Recursively check for problematic float values."""
            import math
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    check_problematic_values(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_problematic_values(item, f"{path}[{i}]")
            elif isinstance(obj, float):
                if math.isnan(obj):
                    logger.warning(f"NaN value found at {path}")
                elif math.isinf(obj):
                    logger.warning(f"{'Positive' if obj > 0 else 'Negative'} infinity found at {path}")
        
        # Check entire result object
        check_problematic_values(result)
        
        # Get the fitness of our evolved solution
        evolved_fitness = result["metadata"]["fitness"]
        logger.info(f"Evolved function fitness: {evolved_fitness}")
        
        # Check for stop before submitting
        if self.should_stop(stop_flag):
            logger.info("Evolution completed but submission cancelled: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered before submission"}
            
        # Only submit if better than the best parent
        if evolved_fitness > best_parent_fitness:
            logger.info(f"Evolved solution fitness ({evolved_fitness}) is better than best parent fitness ({best_parent_fitness}). Submitting.")
            
            serialized_result = serialize_evolution_result(result)
            # Submit the evolved function
            return self.submit_evolution(
                batch_id=batch_id,
                evolved_function=serialized_result["evolved_function"],
                parent_functions=serialized_result["parent_functions"],
                metadata=serialized_result["metadata"]
            )
        else:
            logger.info(f"Evolved solution fitness ({evolved_fitness}) is not better than best parent fitness ({best_parent_fitness}). Not submitting.")
            # Return a result indicating no submission was made
            return {
                "status": "not_submitted",
                "reason": "Evolved solution did not improve upon best parent function",
                "batch_id": batch_id,
                "best_parent_id": best_parent_id,
                "best_parent_fitness": best_parent_fitness,
                "evolved_fitness": evolved_fitness
            }
    
    def process_evaluation_task(self, task_data: Dict, stop_flag=None) -> Dict:
        """
        Process an evaluation task.

        Args:
            task_data: Task data from request_task
            stop_flag: Optional StopFlag object to check for stopping

        Returns:
            Submission response or cancellation status
        """
        # Check for stop before processing
        if self.should_stop(stop_flag):
            logger.info("Evaluation task processing cancelled: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered"}

        # Extract task information
        batch_id = task_data["batch_id"]
        functions = task_data["functions"]

        # Prepare dataset using DatasetFactory for the component type
        component_type = task_data.get("component_type", "loss")
        
        if component_type == "loss_function":
            component_type = "loss"
        elif component_type == "activation_function":
            component_type = "activation"

        component_strategy = ComponentStrategyFactory.create_strategy(component_type)

        # Evaluate each function
        evaluations = []
        for i, function in enumerate(functions):
            # Check for stop signal periodically (every function)
            if self.should_stop(stop_flag):
                logger.info(f"Evaluation cancelled after {i} functions: stop flag triggered")
                if evaluations:
                    # Return partial results if we have any
                    return self.submit_evaluation(batch_id=batch_id, evaluations=evaluations)
                else:
                    return {"status": "cancelled", "reason": "Stop flag triggered during evaluation"}
                    
            function_id = function["id"]
            genetic_code = function["function"]

            # Deserialize genetic code if needed
            if isinstance(genetic_code, str):
                try:
                    genetic_code = deserialize_genetic_code(genetic_code)
                except ValueError:
                    logger.error(f"Error deserializing function {function['id']} - skipping")
                    continue

            # Skip if not a list (improperly formatted genetic code)
            if not isinstance(genetic_code, list):
                logger.error(f"Function {function['id']} is not a valid genetic code - skipping")
                continue

            # Evaluate using component strategy, passing stop_flag
            try:
                score = component_strategy.evaluate(genetic_code, stop_flag=stop_flag)
                
                # Handle if evaluation was cancelled
                if score is None and self.should_stop(stop_flag):
                    logger.info(f"Evaluation of function {function_id} cancelled: stop flag triggered")
                    if evaluations:
                        # Return partial results if we have any
                        return self.submit_evaluation(batch_id=batch_id, evaluations=evaluations)
                    else:
                        return {"status": "cancelled", "reason": "Stop flag triggered during individual evaluation"}

                # Create evaluation record
                evaluations.append({
                "function_id": function_id,
                "score": float(score),
                "criteria": {
                    "accuracy": float(score),
                    "stability": 1.0,
                    "efficiency": 1.0
                    }
                })
                logger.info(f"Evaluated function {function['id']} with score: {score}")
            except Exception as e:
                logger.error(f"Error evaluating function {function['id']}: {e}")
                # Add zero score for failed evaluations
                evaluations.append({
                "function_id": function_id,
                "score": 0.0,
                "criteria": {
                    "accuracy": 0.0,
                    "stability": 0.0,
                    "efficiency": 0.0
                }
                })

        # Check for stop before submission
        if self.should_stop(stop_flag):
            logger.info("Evaluation completed but submission cancelled: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered before submission"}
            
        # Submit the evaluations
        return self.submit_evaluation(
            batch_id=batch_id,
            evaluations=evaluations
        )
    
    def run_mining_cycle(self, task_type: str = "evolve", stop_flag=None) -> Dict:
        """
        Run a complete mining cycle.
        First resets any active tasks, then requests and processes a new task.
        
        Args:
            task_type: Type of task to request ("evolve" or "evaluate")
            stop_flag: Optional StopFlag object to check for stopping
            
        Returns:
            Submission result or None if cancelled
        """
        # Check for stop before starting
        if self.should_stop(stop_flag):
            logger.info("Mining cycle cancelled before starting: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered before cycle started"}
            
        # First reset any active tasks
        try:
            reset_result = self.reset_active_tasks()
            logger.info(f"Reset result: {reset_result}")
        except Exception as e:
            logger.warning(f"Reset failed, but continuing anyway: {e}")
        
        # Check for stop after reset
        if self.should_stop(stop_flag):
            logger.info("Mining cycle cancelled after reset: stop flag triggered")
            return {"status": "cancelled", "reason": "Stop flag triggered after reset"}
            
        # Request a task, passing the stop_flag
        task_data = self.request_task(task_type, stop_flag=stop_flag)
        
        # Check if request was cancelled or failed
        if task_data is None:
            if self.should_stop(stop_flag):
                logger.info("Task request was cancelled")
                return {"status": "cancelled", "reason": "Task request cancelled by stop flag"}
            else:
                logger.error("Task request failed")
                return {"status": "error", "reason": "Task request failed"}
        
        # Process the task, passing the stop_flag
        if task_type == "evolve":
            return self.process_evolution_task(task_data, stop_flag=stop_flag)
        elif task_type == "evaluate":
            return self.process_evaluation_task(task_data, stop_flag=stop_flag)
        else:
            raise ValueError(f"Invalid task type: {task_type}")

    def run_continuous_mining(self, 
                            cycles: int = 0, 
                            alternate: bool = True,
                            delay: float = 5.0,
                            max_retries: int = 3,
                            stop_flag=None):
        """
        Run continuous mining cycles with improved error handling.
        
        Args:
            cycles: Number of cycles to run (0 for infinite)
            alternate: Alternate between evolution and evaluation
            delay: Delay between cycles in seconds
            max_retries: Maximum number of retries for a failed task
            stop_flag: StopFlag object to check for stopping
        """
        cycle_count = 0
        task_type = "evolve"
        
        # Initialize stop flag
        self._stop_mining = False
        
        try:
            # Try to register if not already registered
            try:
                self.register()
                logger.info(f"Registered miner: {self.public_address}")
            except Exception as e:
                if "already registered" not in str(e).lower():
                    raise
                logger.info(f"Miner already registered: {self.public_address}")
            
            # Main mining loop
            while (cycles == 0 or cycle_count < cycles):
                # Check both the internal flag and the shared stop flag
                if self.should_stop(stop_flag):
                    logger.info("Mining stopped by stop flag")
                    return
                    
                retry_count = 0
                success = False
                
                while not success and retry_count < max_retries:
                    # Check stop flag in retry loop too
                    if self.should_stop(stop_flag):
                        logger.info("Mining stopped by stop flag during retry")
                        return
                        
                    try:
                        logger.info(f"Starting {task_type} cycle {cycle_count + 1}...")
                        # Pass stop_flag to run_mining_cycle
                        result = self.run_mining_cycle(task_type, stop_flag=stop_flag)
                        
                        # Check if the task was cancelled by stop flag
                        if isinstance(result, dict) and result.get("status") == "cancelled":
                            logger.info(f"Mining cycle was cancelled: {result.get('reason', 'Unknown reason')}")
                            return  # Exit the mining loop completely
                        
                        # Check if the task was reset
                        if isinstance(result, dict) and result.get("status") == "reset_needed":
                            logger.info("Task was reset. Retrying...")
                            time.sleep(delay)
                            retry_count += 1
                            continue
                        
                        logger.info(f"Completed {task_type} cycle: {result}")
                        success = True
                        
                        # Update task type if alternating
                        if alternate:
                            task_type = "evaluate" if task_type == "evolve" else "evolve"
                        
                        cycle_count += 1
                        
                    except requests.exceptions.HTTPError as e:
                        if e.response and e.response.status_code == 400:
                            try:
                                error_detail = e.response.json().get("detail", "")
                                if "already has active task" in error_detail:
                                    logger.warning(f"Warning: {error_detail}, attempting to reset active tasks...")
                                    self.reset_active_tasks()
                                    logger.info("Active tasks reset, retrying...")
                                    time.sleep(delay)
                                    retry_count += 1
                                    continue
                            except Exception:
                                pass
                        
                        logger.error(f"HTTP Error in mining cycle: {e}")
                        retry_count += 1
                        time.sleep(delay * 2)
                        
                    except Exception as e:
                        logger.error(f"Error in mining cycle: {e}")
                        retry_count += 1
                        time.sleep(delay * 2)
                
                # Check stop flag after cycle completion
                if self.should_stop(stop_flag):
                    logger.info("Mining stopped by stop flag after cycle completion")
                    return
                    
                # Add delay between cycles
                if delay > 0 and success:
                    time.sleep(delay)
                            
        except KeyboardInterrupt:
            logger.info("Mining stopped by user")
        except Exception as e:
            logger.error(f"Error in mining loop: {e}")
        
        logger.info(f"Mining complete. Completed {cycle_count} cycles.")


    def reset_active_tasks(self) -> Dict:
        """
        Reset all active tasks for this miner.
        Use this when you're unable to request new tasks due to 'already has active task' errors.
        
        Returns:
            Response data
        """
        self._ensure_session()
        
        auth_data = self._create_signature_data()
        response = self.session.post(
            f"{self.base_url}{self.api_prefix}/tasks/{self.public_address}/reset_active",
            json=auth_data
        )
        response.raise_for_status()
        return response.json()

class BittensorPoolClient(PoolClient):
    """Client for interacting with the Neural Component Pool server using Bittensor wallets."""
    
    def __init__(self, 
                 wallet: Optional[Wallet] = None,
                 base_url: str = "http://localhost:8000"):
        """
        Initialize the client with a bittensor wallet.
        
        Args:
            wallet: Bittensor wallet object
            base_url: Base URL for the API server
        """
        if not wallet:
            raise ValueError("Wallet must be provided")
            
        self.wallet = wallet
        
        # Initialize the parent class with the wallet's hotkey address
        super(BittensorPoolClient, self).__init__(
            public_address=wallet.hotkey.ss58_address,
            base_url=base_url
        )
    
    def _create_signature_data(self) -> Dict:
        """
        Create authentication data for API requests.
        
        Returns:
            Dictionary with public_address, signature, and message
        """
        # Create a timestamped message to prevent replay attacks
        message = f"auth:{int(time.time())}"
        signature = self.wallet.hotkey.sign(message).hex()
        
        auth_data = {
            "public_address": self.public_address,
            "signature": signature,
            "message": message
        }
        
        logger.debug(f"Auth data: {auth_data}")
        return auth_data

# Example usage
def main():
    # Create a bittensor wallet
    wallet = wallet(name="default", hotkey="default")
    
    with BittensorPoolClient(wallet=wallet, base_url="http://localhost:8000") as client:
        # Register miner
        client.register()
        
        # Run 10 mining cycles, alternating between evolution and evaluation
        client.run_continuous_mining(cycles=10, alternate=True)

if __name__ == "__main__":
    main()