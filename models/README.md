# Attention Models

This directory contains implementations of various attention mechanisms used in deep learning.

## 📁 Available Models

### 1. [Self Attention](selfhead/README.md)
- **File**: `selfhead.py`
- **Description**: Basic self-attention mechanism where each position attends to all positions in the sequence
- **Key Features**: Single attention head, self-referential attention
- **Best For**: Learning attention fundamentals, simple sequence modeling

### 2. [Multi-Head Attention](multihead/README.md)
- **File**: `multihead.py`
- **Description**: Advanced attention mechanism with multiple parallel attention heads
- **Key Features**: Multiple attention heads, parallel computation, enhanced representation
- **Best For**: Transformer architectures, complex sequence modeling, production systems

## 🚀 Quick Start

### Using the Training Interface

```bash
# Train self attention
python main.py --model selfhead --embed_dim 64 --epochs 100

# Train multi-head attention
python main.py --model multihead --embed_dim 128 --num_heads 8 --epochs 200

# Interactive mode
python main.py --interactive
```

### Direct Model Usage

```python
# Self Attention
from models.selfhead import SelfAttention
model = SelfAttention(emd_dim=64)

# Multi-Head Attention
from models.multihead import MultiAttention
model = MultiAttention(emd_dim=64, num_heads=8)
```

## 📊 Model Comparison

| Feature | Self Attention | Multi-Head Attention |
|---------|----------------|---------------------|
| **Heads** | 1 | Multiple (configurable) |
| **Complexity** | O(n²) | O(n² × h) |
| **Memory** | Lower | Higher |
| **Representation** | Single pattern | Multiple patterns |
| **Use Case** | Learning, simple tasks | Production, complex tasks |

## 🔗 Navigation

- **[Main Documentation](../README.md)** - Project overview and attention basics
- **[Self Attention](selfhead/README.md)** - Detailed self-attention documentation
- **[Multi-Head Attention](multihead/README.md)** - Detailed multi-head attention documentation

## 🤝 Contributing

To add a new attention model:

1. Create a new directory: `models/your_model/`
2. Add your implementation: `models/your_model/your_model.py`
3. Create documentation: `models/your_model/README.md`
4. Update this index file
5. Update the main training interface in `main.py`

## 📝 Model Requirements

Each attention model should:

- Inherit from `torch.nn.Module`
- Implement a `forward(x)` method
- Accept appropriate parameters in `__init__`
- Return output with the same shape as input
- Include proper documentation

---

**Start with [Self Attention](selfhead/README.md) for fundamentals, then explore [Multi-Head Attention](multihead/README.md) for advanced usage!** 