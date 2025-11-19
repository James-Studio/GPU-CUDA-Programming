# GPU CUDA Programming - Autograd Tensor Library

A minimal autograd tensor library implemented in Python with CUDA acceleration, similar to PyTorch. This project provides automatic differentiation capabilities for building and training neural networks on GPU.

## Overview

This project implements an autograd tensor library (`AGTensor`) that wraps CUDA-accelerated tensor operations with automatic differentiation. It includes:

- **Python bindings** for CUDA tensor operations using pybind11
- **Automatic differentiation** via reverse-mode backpropagation
- **GPU acceleration** for tensor computations
- **Neural network training** support with MNIST dataset example

## Features

### Supported Operations

- **Elementwise operations**: `+`, `-`, `*` (with broadcasting)
- **Matrix operations**: `@` (matrix multiplication), transpose
- **Activation functions**: ReLU
- **Reduction operations**: `sum`, `mean`
- **Loss functions**: Cross-entropy loss
- **Non-differentiable operations**: `argmax`, `==` (equality comparison)

### Key Components

- `AGTensor`: Autograd wrapper class that tracks computation graphs
- CUDA tensor operations: Efficient GPU-accelerated computations
- Backward propagation: Automatic gradient computation through the computation graph

## Requirements

- **CUDA Toolkit** (for CUDA support)
- **CMake** (version 3.20 or higher)
- **Python 3** with development headers
- **pybind11** (automatically downloaded by CMake)
- **NumPy**
- **PyTorch** (for testing and MNIST dataset loading)
- **pytest** (for running tests)

## Building

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure and build:
```bash
cmake ..
make
```

This will:
- Download pybind11 if not found
- Compile CUDA tensor operations
- Build Python bindings as the `bten` module

## Usage

### Basic Example

```python
import numpy as np
from mygrad.engine import AGTensor

# Create an AGTensor from a NumPy array
a_np = np.arange(1, 4, dtype=np.float32).reshape(1, 3)
a = AGTensor(a_np, is_cuda=True)

# Perform operations
b = a + 3.0
c = b * 2.0

# Compute gradients
c.backward()

# Access gradients
print(a.grad.to_numpy())  # Print gradients of a
```

### Training a Neural Network

Train a 2-layer MLP on the MNIST dataset:

```bash
python train_mnist.py
```

This will:
- Load the MNIST dataset
- Initialize a 2-layer MLP (784 → 128 → 10)
- Train for 20 epochs using SGD
- Report training loss and test accuracy

Example output:
```
Epoch 1: train loss = 0.7655 (latency 16.58s)
  Test acc: 0.9022
Epoch 2: train loss = 0.3108 (latency 16.29s)
  Test acc: 0.9245
...
```

## Testing

Run the test suite to verify your implementation:

```bash
pytest -v test_ag_tensor.py
```

To run a specific test:

```bash
pytest -v test_ag_tensor.py::test_mul
```

The test suite includes tests for:
- Sum reduction
- Transpose
- Elementwise operations (add, sub, mul)
- Matrix multiplication
- ReLU activation
- Cross-entropy loss
- Argmax
- Equality comparison
- 2-layer MLP training

## Project Structure

```
.
├── CMakeLists.txt          # Build configuration
├── src/
│   ├── bindings.cu         # Python bindings using pybind11
│   ├── py_tensor_shim.hh   # PyTensor wrapper class
│   └── ops/                # CUDA operation implementations
│       ├── op_cross_entropy.cuh
│       ├── op_elemwise.cuh
│       ├── op_mm.cuh
│       └── op_reduction.cuh
├── utils/                  # Utility headers
│   ├── check_error.cuh
│   ├── dataset_mnist.hh
│   └── tensor.cuh
├── mygrad/
│   ├── __init__.py
│   └── engine.py           # AGTensor implementation
├── test_ag_tensor.py       # Test suite
├── train_mnist.py          # MNIST training script
└── lab-2.md                # Lab instructions
```

## Implementation Details

### AGTensor Class

The `AGTensor` class wraps CUDA tensors and tracks:
- **Data**: The underlying tensor (CPU or GPU)
- **Gradients**: Computed gradients during backpropagation
- **Computation graph**: Parent tensors and backward functions
- **Gradient tracking**: `requires_grad` flag to enable/disable autograd

### Backward Propagation

The `backward()` method:
1. Traverses the computation graph in reverse topological order
2. Calls each operation's backward function
3. Accumulates gradients for each tensor in the graph

### GPU Acceleration

Tensors can be stored on GPU by setting `is_cuda=True`. All operations are performed on the device where the tensor resides, providing significant speedup for large computations.

## Notes

- The Python autograd wrapper incurs some overhead compared to pure C++ implementations
- For best performance, ensure CUDA-compatible GPU is available
- The framework performs automatic memory management for GPU tensors

## License

This project is part of a GPU/CUDA programming course lab assignment.

