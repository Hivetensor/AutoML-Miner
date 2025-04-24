"""Dataset factory for creating standard machine learning datasets."""

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any
import logging
import os
import gzip
import pickle
try:
    from torchvision import datasets, transforms
    torchvision_available = True
except ImportError:
    torchvision_available = False

logger = logging.getLogger(__name__)

class DatasetFactory:
    """Factory class for creating standard machine learning datasets.
    
    This class provides static methods to load or generate datasets for model 
    training and evaluation, with a consistent interface returning trainX, 
    trainY, valX, valY keys.
    """
    
    _cached_datasets = {}  # Cache for loaded datasets
    
    @staticmethod
    def create_dataset(dataset_name: str = "mnist", config: Optional[Dict[str, Any]] = None, eval_seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Create or load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load (e.g., "mnist", "cifar10", "tiny")
            config: Optional configuration dictionary to customize the dataset loading
        
        Returns:
            A dictionary with keys 'trainX', 'trainY', 'valX', 'valY'
        """
        if not dataset_name:
            dataset_name = "mnist"  # Default to MNIST if not specified
            
        config = config or {}
        logger.info(f"Creating dataset: {dataset_name}")
        
        # Normalize dataset name to lowercase
        dataset_name = dataset_name.lower()
        
        # Look for dataset in cache
        cache_key = f"{dataset_name}_{hash(str(config))}_{eval_seed}"
        if cache_key in DatasetFactory._cached_datasets:
            logger.info(f"Using cached dataset: {dataset_name}")
            return DatasetFactory._cached_datasets[cache_key]
            
        # Choose dataset creation method based on name
        if dataset_name == "mnist":
            dataset = DatasetFactory._load_mnist(config, seed=eval_seed)
        elif dataset_name == "fashion_mnist":
            dataset = DatasetFactory._load_fashion_mnist(config, seed=eval_seed)
        elif dataset_name == "cifar10":
            dataset = DatasetFactory._load_cifar10(config, seed=eval_seed)
        else:
            logger.warning(f"Unknown dataset '{dataset_name}', falling back to MNIST")
            dataset = DatasetFactory._load_mnist(config, seed=eval_seed)
        
        # Cache the dataset unless configured not to
        if not config.get("no_cache", False):
            DatasetFactory._cached_datasets[cache_key] = dataset
            
        return dataset
    
    @staticmethod
    def _load_mnist(config: Dict[str, Any] = None, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Load the MNIST dataset.
        
        Args:
            config: Configuration dictionary with optional keys:
                   - 'sample_size': Number of examples to load (default: use all)
                   - 'val_split': Fraction of training data to use for validation (default: 0.2)
                   - 'flatten': Whether to flatten images to 1D (default: True)
                   - 'normalize': Whether to normalize pixel values (default: True)
                   - 'one_hot': Whether to use one-hot encoding for labels (default: True)
                   - 'data_path': Path to look for MNIST data (default: './data')
                   
        Returns:
            Dictionary with keys 'trainX', 'trainY', 'valX', 'valY'
        """
        config = config or {}
        sample_size = config.get("sample_size", 2000)  # None means use all
        val_split = config.get("val_split", 0.1)
        flatten = config.get("flatten", True)
        normalize = config.get("normalize", True)
        one_hot = config.get("one_hot", True)
        data_path = config.get("data_path", './data')
        
        logger.info("Loading MNIST dataset")
        
        # Try to load using torchvision if available
        if torchvision_available:
            try:
                # Define transformations
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
                ])
                
                # Load training data
                train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
                
                # Convert to tensors
                trainX = []
                trainY = []
                
                # Use dataloader to efficiently load data
                def seed_worker(worker_id):
                    worker_seed = seed if seed is not None else torch.initial_seed() % 2**32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                    
                # Create DataLoader with fixed seed
                g = torch.Generator()
                if seed is not None:
                    g.manual_seed(seed)
                    
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset) if sample_size is None else sample_size,
                    worker_init_fn=seed_worker,
                    generator=g
                )

                for data, target in train_loader:
                    trainX = data
                    trainY = target
                    break  # Only need one batch since we loaded everything
                
                # Flatten if required
                if flatten:
                    trainX = trainX.reshape(trainX.shape[0], -1)
                
                # Convert to one-hot if required
                if one_hot:
                    num_classes = 10
                    trainY_one_hot = torch.zeros(trainY.size(0), num_classes)
                    trainY_one_hot.scatter_(1, trainY.unsqueeze(1), 1)
                    trainY = trainY_one_hot
                    
                # Split into train and validation
                val_size = int(trainX.size(0) * val_split)
                train_size = trainX.size(0) - val_size
                
                # Randomly shuffle the data
                if seed is not None:
                    # Set seed for reproducible splitting
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    import random
                    random.seed(seed)
                    logger.info(f"Using seed {seed} for deterministic MNIST dataset")

                    # If using CUDA, also set its seed
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        
                    # Force deterministic operations in PyTorch
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                indices = torch.randperm(trainX.size(0))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                
                valX = trainX[val_indices]
                valY = trainY[val_indices]
                trainX = trainX[train_indices]
                trainY = trainY[train_indices]
                
                logger.info(f"Successfully loaded MNIST with torchvision: "
                          f"trainX: {trainX.shape}, trainY: {trainY.shape}, "
                          f"valX: {valX.shape}, valY: {valY.shape}")
                          
                return {
                    "trainX": trainX,
                    "trainY": trainY,
                    "valX": valX,
                    "valY": valY
                }
                
            except Exception as e:
                logger.warning(f"Failed to load MNIST with torchvision: {e}")
                # Fall back to manual loading
        
        # Manual loading if torchvision failed or isn't available
        try:
            # Paths for the MNIST files
            train_images_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
            train_labels_path = os.path.join(data_path, 'train-labels-idx1-ubyte.gz')
            
            # Check if files exist
            if not os.path.exists(train_images_path) or not os.path.exists(train_labels_path):
                logger.error(f"MNIST data files not found at {data_path}")
                # Fall back to tiny dataset
                raise ValueError("Failing to get MNIST files")

            # Load the MNIST data
            with gzip.open(train_images_path, 'rb') as f:
                # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
                header = f.read(16)
                buf = f.read()
                images = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)
                if sample_size is not None:
                    images = images[:sample_size]
                
            with gzip.open(train_labels_path, 'rb') as f:
                # First 8 bytes are magic_number, n_labels
                header = f.read(8)
                buf = f.read()
                labels = np.frombuffer(buf, dtype=np.uint8)
                if sample_size is not None:
                    labels = labels[:sample_size]
            
            # Convert to PyTorch tensors
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels)
            
            # Flatten if needed
            if flatten:
                images = images.reshape(images.shape[0], -1)
            
            # Normalize if needed
            if normalize:
                images = images.float() / 255.0
            
            # Convert labels to one-hot if needed
            if one_hot:
                num_classes = 10
                labels_one_hot = torch.zeros(labels.size(0), num_classes)
                labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                labels = labels_one_hot
            
            # Split into train and validation
            val_size = int(images.size(0) * val_split)
            train_size = images.size(0) - val_size
            
            # Randomly shuffle the data
            # Replace random shuffling with seeded version
            if seed is not None:
                # Set seed for reproducible splitting
                torch.manual_seed(seed)
                np.random.seed(seed)
                import random
                random.seed(seed)
                logger.info(f"Using seed {seed} for deterministic MNIST dataset")

                # If using CUDA, also set its seed
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    
                # Force deterministic operations in PyTorch
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


            indices = torch.randperm(images.size(0))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            
            trainX = images[train_indices]
            trainY = labels[train_indices]
            valX = images[val_indices]
            valY = labels[val_indices]
            
            logger.info(f"Successfully loaded MNIST manually: "
                      f"trainX: {trainX.shape}, trainY: {trainY.shape}, "
                      f"valX: {valX.shape}, valY: {valY.shape}")
            
            return {
                "trainX": trainX,
                "trainY": trainY,
                "valX": valX,
                "valY": valY
            }
            
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            # Fall back to tiny dataset
            raise ValueError("Failed to load dataset")
            
    
    @staticmethod
    def _load_fashion_mnist(config: Dict[str, Any] = None, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Load the Fashion-MNIST dataset.
        
        Similar to MNIST but with fashion items instead of digits.
        See _load_mnist for config options.
        """
        config = config or {}
        
        if torchvision_available:
            try:
                config_copy = config.copy()
                data_path = config_copy.get("data_path", './data')
                
                # Define transformations similar to MNIST
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,)) if config_copy.get("normalize", True) else transforms.Lambda(lambda x: x)
                ])
                
                # Load Fashion-MNIST specifically
                train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
                
                # Rest of processing is identical to MNIST
                # This is a simplified version - in practice, use code similar to _load_mnist
                # Converting to numpy for processing

                def seed_worker(worker_id):
                    worker_seed = seed if seed is not None else torch.initial_seed() % 2**32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                    
                # Create DataLoader with fixed seed
                g = torch.Generator()
                if seed is not None:
                    g.manual_seed(seed)
                    
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset) if config_copy.get("sample_size") is None else config_copy.get("sample_size"),
                    worker_init_fn=seed_worker,
                    generator=g
                )
                
                # Get the data
                for data, target in train_loader:
                    trainX = data
                    trainY = target
                    break
                
                # Process according to config
                if config_copy.get("flatten", True):
                    trainX = trainX.reshape(trainX.shape[0], -1)
                
                if config_copy.get("one_hot", True):
                    num_classes = 10
                    trainY_one_hot = torch.zeros(trainY.size(0), num_classes)
                    trainY_one_hot.scatter_(1, trainY.unsqueeze(1), 1)
                    trainY = trainY_one_hot
                
                # Split into train and val
                val_split = config_copy.get("val_split", 0.2)
                val_size = int(trainX.size(0) * val_split)
                train_size = trainX.size(0) - val_size
                
                # Shuffle for randomness
                # Randomly shuffle the data
                if seed is not None:
                    # Set seed for reproducible splitting
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    import random
                    random.seed(seed)
                    logger.info(f"Using seed {seed} for deterministic MNIST dataset")

                    # If using CUDA, also set its seed
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        
                    # Force deterministic operations in PyTorch
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                indices = torch.randperm(trainX.size(0))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                
                valX = trainX[val_indices]
                valY = trainY[val_indices]
                trainX = trainX[train_indices]
                trainY = trainY[train_indices]
                
                logger.info(f"Successfully loaded Fashion-MNIST: "
                          f"trainX: {trainX.shape}, trainY: {trainY.shape}, "
                          f"valX: {valX.shape}, valY: {valY.shape}")
                
                return {
                    "trainX": trainX,
                    "trainY": trainY,
                    "valX": valX,
                    "valY": valY
                }
                
            except Exception as e:
                logger.warning(f"Failed to load Fashion-MNIST: {e}")
                # Fall back to MNIST
                logger.info("Falling back to MNIST dataset")
                return DatasetFactory._load_mnist(config)
        else:
            logger.warning("torchvision not available, falling back to MNIST")
            return DatasetFactory._load_mnist(config)
    
    @staticmethod
    def _load_cifar10(config: Dict[str, Any] = None, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Load the CIFAR-10 dataset.
        
        Args:
            config: Similar to _load_mnist but with CIFAR-specific defaults
                   
        Returns:
            Dictionary with keys 'trainX', 'trainY', 'valX', 'valY'
        """
        config = config or {}
        
        if torchvision_available:
            try:
                config_copy = config.copy()
                data_path = config_copy.get("data_path", './data')
                
                # Define CIFAR transformations
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) 
                    if config_copy.get("normalize", True) else transforms.Lambda(lambda x: x)
                ])
                
                # Load CIFAR-10
                train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
                
                # Get the data
                def seed_worker(worker_id):
                    worker_seed = seed if seed is not None else torch.initial_seed() % 2**32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                    
                # Create DataLoader with fixed seed
                g = torch.Generator()
                if seed is not None:
                    g.manual_seed(seed)
                    
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset) if config_copy.get("sample_size") is None else config_copy.get("sample_size"),
                    worker_init_fn=seed_worker,
                    generator=g
                )


                for data, target in train_loader:
                    trainX = data
                    trainY = target
                    break
                
                # Flatten if needed (note: CIFAR is 32x32x3)
                if config_copy.get("flatten", True):
                    trainX = trainX.reshape(trainX.shape[0], -1)
                
                # One-hot encode
                if config_copy.get("one_hot", True):
                    num_classes = 10
                    trainY_one_hot = torch.zeros(trainY.size(0), num_classes)
                    trainY_one_hot.scatter_(1, trainY.unsqueeze(1), 1)
                    trainY = trainY_one_hot
                
                # Split train/val
                val_split = config_copy.get("val_split", 0.2)
                val_size = int(trainX.size(0) * val_split)
                train_size = trainX.size(0) - val_size

                # Replace random shuffling with seeded version
                if seed is not None:
                    # Set seed for reproducible splitting
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    import random
                    random.seed(seed)
                    logger.info(f"Using seed {seed} for deterministic MNIST dataset")

                    # If using CUDA, also set its seed
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        
                    # Force deterministic operations in PyTorch
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                
                indices = torch.randperm(trainX.size(0))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                
                valX = trainX[val_indices]
                valY = trainY[val_indices]
                trainX = trainX[train_indices]
                trainY = trainY[train_indices]
                
                logger.info(f"Successfully loaded CIFAR-10: "
                          f"trainX: {trainX.shape}, trainY: {trainY.shape}, "
                          f"valX: {valX.shape}, valY: {valY.shape}")
                
                return {
                    "trainX": trainX,
                    "trainY": trainY,
                    "valX": valX,
                    "valY": valY
                }
                
            except Exception as e:
                logger.warning(f"Failed to load CIFAR-10: {e}")
                # Fall back to MNIST
                logger.info("Falling back to MNIST dataset")
                return DatasetFactory._load_mnist(config)
        else:
            logger.warning("torchvision not available, falling back to MNIST")
            return DatasetFactory._load_mnist(config)
    
