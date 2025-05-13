# Neural Component Pool Mining Client

A distributed evolutionary computing system for discovering and optimizing neural network loss functions through genetic programming.

## Overview

This project enables participation in a distributed evolutionary network that evolves neural network components (specifically loss functions) using genetic programming techniques. The system allows miners to contribute computing resources to either:

1. **Evolve** - Create new loss functions through genetic operators
2. **Evaluate** - Test the performance of evolved functions on standard datasets

The best-performing functions are rewarded with Alpha tokens in the Neural Component Pool ecosystem.

## Features

- Participation in distributed evolutionary computation
- Support for both evolution and evaluation tasks
- Genetic programming framework for loss function representation
- Multiple evaluation strategies for fitness assessment
- Cryptographic identity management via substrate keypairs
- Simple command-line interface for miners

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0.0+
- substrate-interface (for cryptographic operations)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/neural-component-pool.git
   cd neural-component-pool
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Install additional dependencies:
   ```bash
   pip install substrate-interface aiohttp asyncio
   ```

## Usage

The mining client can be started with various options:

```bash
python run-miner.py [options]
```

### Authentication Options

The client uses substrate cryptography for secure authentication:

```bash
# Use a mnemonic phrase directly
python run-miner.py --mnemonic "your twelve word mnemonic seed phrase goes here example test"

# OR use a file containing a mnemonic or seed
python run-miner.py --seed-file /path/to/seed.txt
```

If no key is provided, a new one will be generated.

### Server Configuration

```bash
# Connect to a specific API server
python run-miner.py --api-url "http://pool-server.example.com:8000"
```

### Mining Options

```bash
# Run evolution tasks
python run-miner.py --task-type evolve

# Run evaluation tasks
python run-miner.py --task-type evaluate

# Alternate between evolution and evaluation
python run-miner.py --alternate

# Run for a specific number of cycles
python run-miner.py --cycles 100

# Set delay between mining cycles
python run-miner.py --delay 10.0
```

### Full Example

```bash
python run-miner.py --seed-file ~/my_miner_key.txt --pool_url "http://pool.example.com:8000" --alternate --cycles 0 --delay 5.0
python run-miner.py --pool_url "http://localhost:8080" --alternate --cycles 0 --delay 10.0
```

## How It Works

### Genetic Programming Framework

The system represents loss functions as genetic programs - sequences of instructions that can be evolved using genetic operators:

1. **Genetic Code**: Loss functions are represented as lists of instructions
2. **Interpreter**: Executes genetic code on PyTorch tensors
3. **Evolution**: New functions are created through mutation and crossover
4. **Evaluation**: Functions are assessed by training small neural networks

### Mining Process

1. **Registration**: Miner registers with the pool using their cryptographic identity
2. **Task Request**: Miner requests a task (evolution or evaluation)
3. **Computation**: Task is performed locally using the miner's resources
4. **Submission**: Results are submitted back to the pool with cryptographic proof
5. **Rewards**: Miners receive Alpha tokens based on contribution

## System Architecture

The system consists of several components:

- **Client API**: Interface for connecting to the Neural Component Pool server
- **Evolution Engine**: Manages genetic algorithm for function evolution
- **Evaluation Strategies**: Different methods for assessing function fitness
- **Genetic Operators**: Tools for mutation, crossover, and selection
- **Interpreter**: Executes genetic code in PyTorch environment
- **Mining Client**: CLI interface for participation

## Advanced Configuration

For advanced users, the system supports customizing:

- Evolution parameters (population size, mutation rate, etc.)
- Evaluation strategies (different optimizers, metrics, etc.)
- Custom datasets for evaluation
- Function serialization for sharing evolved functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.