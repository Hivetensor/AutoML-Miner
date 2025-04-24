"""Test file for the DatasetFactory class."""

import torch
import numpy as np
from automl_client.data.dataset_factory import DatasetFactory


def test_loss_dataset():
    """Test the loss dataset creation."""
    dataset = DatasetFactory.create_dataset("loss")
    
    # Check that the dataset has the expected keys
    assert "predictions" in dataset
    assert "targets" in dataset
    
    # Check that the tensors have the expected shapes
    assert dataset["predictions"].shape == (32, 10)  # Default batch_size=32, num_classes=10
    assert dataset["targets"].shape == (32, 10)  # Default one-hot encoded targets
    
    # Test with custom config
    custom_dataset = DatasetFactory.create_dataset("loss", {
        "batch_size": 16,
        "num_classes": 5,
        "target_type": "class_indices"
    })
    
    assert custom_dataset["predictions"].shape == (16, 5)
    assert custom_dataset["targets"].shape == (16,)  # Class indices shape


def test_optimizer_dataset():
    """Test the optimizer dataset creation."""
    dataset = DatasetFactory.create_dataset("optimizer")
    
    # Check that the dataset has the expected keys
    assert "parameters" in dataset
    assert "gradients" in dataset
    assert "loss" in dataset
    
    # Check that parameters and gradients are lists of tensors
    assert isinstance(dataset["parameters"], list)
    assert isinstance(dataset["gradients"], list)
    
    # Check that loss is a scalar tensor
    assert dataset["loss"].dim() == 0
    
    # Test with custom config
    custom_dataset = DatasetFactory.create_dataset("optimizer", {
        "num_params": 200,
        "num_layers": 2
    })
    
    assert len(custom_dataset["parameters"]) == 2  # num_layers
    assert len(custom_dataset["gradients"]) == 2  # num_layers


def test_activation_dataset():
    """Test the activation dataset creation."""
    dataset = DatasetFactory.create_dataset("activation")
    
    # Check that the dataset has the expected keys
    assert "inputs" in dataset
    
    # Check that the tensor has the expected shape
    assert dataset["inputs"].shape == (32, 64)  # Default batch_size=32, feature_size=64
    
    # Test with custom config
    custom_dataset = DatasetFactory.create_dataset("activation", {
        "batch_size": 8,
        "feature_size": 32,
        "range": (-10, 10)
    })
    
    assert custom_dataset["inputs"].shape == (8, 32)
    # Check that values are in the expected range
    assert torch.all(custom_dataset["inputs"] >= -10)
    assert torch.all(custom_dataset["inputs"] <= 10)


def test_regularization_dataset():
    """Test the regularization dataset creation."""
    dataset = DatasetFactory.create_dataset("regularization")
    
    # Check that the dataset has the expected keys
    assert "weights" in dataset
    
    # Check that weights is a list of tensors
    assert isinstance(dataset["weights"], list)
    assert len(dataset["weights"]) == 3  # Default num_layers=3
    
    # Test with custom config
    custom_dataset = DatasetFactory.create_dataset("regularization", {
        "num_params": 500,
        "num_layers": 4,
        "sparsity": 0.5
    })
    
    assert len(custom_dataset["weights"]) == 4  # num_layers


def test_default_dataset():
    """Test the default dataset creation."""
    dataset = DatasetFactory.create_dataset("unknown_type")
    
    # Check that the dataset has the expected keys
    assert "data" in dataset
    
    # Check that the tensor has the expected shape
    assert dataset["data"].shape == (10, 10)  # Default shape
    
    # Test with custom config
    custom_dataset = DatasetFactory.create_dataset("unknown_type", {
        "shape": (5, 5)
    })
    
    assert custom_dataset["data"].shape == (5, 5)


def test_dataset_determinism():
    """Test that datasets are deterministically generated with the same seed."""
    # Test for MNIST dataset
    seed1 = 42
    seed2 = 123
    
    # Create datasets with same seed
    mnist_dataset1 = DatasetFactory.create_dataset("mnist", eval_seed=seed1)
    # Clear cache to ensure we're not just getting the cached dataset
    DatasetFactory._cached_datasets = {}
    mnist_dataset2 = DatasetFactory.create_dataset("mnist", eval_seed=seed1)
    
    # Datasets with same seed should be identical
    assert torch.all(torch.eq(mnist_dataset1["trainX"], mnist_dataset2["trainX"]))
    assert torch.all(torch.eq(mnist_dataset1["trainY"], mnist_dataset2["trainY"]))
    assert torch.all(torch.eq(mnist_dataset1["valX"], mnist_dataset2["valX"]))
    assert torch.all(torch.eq(mnist_dataset1["valY"], mnist_dataset2["valY"]))
    
    # Create dataset with different seed
    mnist_dataset3 = DatasetFactory.create_dataset("mnist", eval_seed=seed2)
    
    # Datasets with different seeds should be different
    # Check if at least some elements are different (it's highly unlikely that they would be identical by chance)
    assert not torch.all(torch.eq(mnist_dataset1["trainX"], mnist_dataset3["trainX"]))
    
    # Test for Fashion-MNIST dataset
    fashion_dataset1 = DatasetFactory.create_dataset("fashion_mnist", eval_seed=seed1)
    DatasetFactory._cached_datasets = {}
    fashion_dataset2 = DatasetFactory.create_dataset("fashion_mnist", eval_seed=seed1)
    
    # Datasets with same seed should be identical
    assert torch.all(torch.eq(fashion_dataset1["trainX"], fashion_dataset2["trainX"]))
    assert torch.all(torch.eq(fashion_dataset1["trainY"], fashion_dataset2["trainY"]))
    assert torch.all(torch.eq(fashion_dataset1["valX"], fashion_dataset2["valX"]))
    assert torch.all(torch.eq(fashion_dataset1["valY"], fashion_dataset2["valY"]))
    
    # Create dataset with different seed
    fashion_dataset3 = DatasetFactory.create_dataset("fashion_mnist", eval_seed=seed2)
    
    # Datasets with different seeds should be different
    assert not torch.all(torch.eq(fashion_dataset1["trainX"], fashion_dataset3["trainX"]))
    
    # Test for CIFAR10 dataset
    cifar_dataset1 = DatasetFactory.create_dataset("cifar10", eval_seed=seed1)
    DatasetFactory._cached_datasets = {}
    cifar_dataset2 = DatasetFactory.create_dataset("cifar10", eval_seed=seed1)
    
    # Datasets with same seed should be identical
    assert torch.all(torch.eq(cifar_dataset1["trainX"], cifar_dataset2["trainX"]))
    assert torch.all(torch.eq(cifar_dataset1["trainY"], cifar_dataset2["trainY"]))
    assert torch.all(torch.eq(cifar_dataset1["valX"], cifar_dataset2["valX"]))
    assert torch.all(torch.eq(cifar_dataset1["valY"], cifar_dataset2["valY"]))
    
    # Create dataset with different seed
    cifar_dataset3 = DatasetFactory.create_dataset("cifar10", eval_seed=seed2)
    
    # Datasets with different seeds should be different
    assert not torch.all(torch.eq(cifar_dataset1["trainX"], cifar_dataset3["trainX"]))


if __name__ == "__main__":
    print("Testing DatasetFactory...")
    test_loss_dataset()
    test_optimizer_dataset()
    test_activation_dataset()
    test_regularization_dataset()
    test_default_dataset()
    test_dataset_determinism()
    print("All tests passed!")
