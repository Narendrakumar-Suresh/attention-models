# Attention Mechanisms in Deep Learning

This repository provides implementations and training interfaces for various attention mechanisms used in deep learning, particularly in transformer architectures.

## 📚 Understanding Attention Mechanisms

### What is Attention?

Attention is a mechanism that allows neural networks to focus on specific parts of the input when making predictions. Think of it like how humans pay attention to different parts of a sentence when understanding its meaning.

### Key Concepts

#### 1. **Query, Key, Value (QKV) Framework**
- **Query (Q)**: What we're looking for
- **Key (K)**: What we're matching against
- **Value (V)**: What we actually want to retrieve

The attention mechanism computes how much attention to pay to each part of the input by comparing queries with keys and then using the values.

#### 2. **Attention Score Calculation**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```
> [!NOTE]  
> Here K^T refers to transpose of matrix K

Where:
- `QK^T` computes the similarity between queries and keys
- `√d_k` is a scaling factor to prevent softmax saturation
- `softmax` converts scores to probabilities
- The result is weighted by values

#### 3. **Why Attention Works**
- **Parallelization**: Unlike RNNs, attention can process all positions simultaneously
- **Long-range dependencies**: Can directly connect any two positions
- **Interpretability**: Attention weights show which parts of input are important

## 🚀 Available Models

### 1. [Self Attention](models/selfhead/README.md)
A basic attention mechanism where each position attends to all positions in the sequence, including itself.

**Key Features:**
- Single attention head
- Self-referential attention
- Foundation for more complex attention mechanisms

### 2. [Multi-Head Attention](models/multihead/README.md)
An advanced attention mechanism that uses multiple attention heads in parallel, allowing the model to attend to different types of information simultaneously.

**Key Features:**
- Multiple attention heads
- Parallel attention computation
- Enhanced representation learning

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Narendrakumar-Suresh/attention-models.git
cd attention-models

# Install dependencies using uv
uv sync
```

### Basic Usage

```bash
# Train with default parameters
uv run main.py

# Interactive mode for parameter selection
uv run main.py --interactive

# Custom training
uv run main.py --model multihead --embed_dim 128 --num_heads 8 --epochs 200
```

### Command Line Options

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--model` | Model type (`multihead` or `selfhead`) | `multihead` | No |
| `--embed_dim` | Embedding dimension | `64` | No |
| `--num_heads` | Number of attention heads (multihead only) | `8` | No |
| `--learning_rate` | Learning rate | `0.001` | No |
| `--batch_size` | Batch size | `32` | No |
| `--seq_length` | Sequence length | `10` | No |
| `--epochs` | Training epochs | `100` | No |
| `--save_path` | Model save path | `None` | No |
| `--interactive` | Interactive parameter selection | `False` | No |

## 📖 Model Documentation

- **[Self Attention](models/selfhead/README.md)** - Basic self-attention implementation
- **[Multi-Head Attention](models/multihead/README.md)** - Multi-head attention with parallel heads

## 🧪 Training Examples

### Example 1: Basic Self Attention
```bash
uv run main.py --model selfhead --embed_dim 32 --epochs 50
```

### Example 2: Multi-Head Attention with Custom Parameters
```bash
uv run main.py --model multihead --embed_dim 256 --num_heads 16 --learning_rate 0.0001 --epochs 300
```

### Example 3: Save Trained Model
```bash
uv run main.py --model multihead --embed_dim 128 --num_heads 8 --save_path models/my_trained_model.pth
```

## 🔬 Understanding the Code

### Project Structure
```
attention/
├── main.py              # Training interface and main entry point
├── models/
│   ├── selfhead.py      # Self attention implementation
│   ├── selfhead/        # Self attention documentation
│   ├── multihead.py     # Multi-head attention implementation
│   └── multihead/       # Multi-head attention documentation
├── pyproject.toml       # Project dependencies
└── README.md           # This file
```

### Key Classes

#### `AttentionTrainer`
The main training class that handles:
- Model initialization
- Training loop
- Evaluation
- Model saving/loading

#### `SelfAttention`
Basic self-attention implementation with:
- Query, Key, Value projections
- Scaled dot-product attention
- Output projection

#### `MultiAttention`
Multi-head attention with:
- Multiple attention heads
- Parallel attention computation
- Concatenation and projection

## 🎯 Use Cases

### When to Use Self Attention
- Simple sequence modeling tasks
- Learning basic attention patterns
- Educational purposes
- Small-scale experiments

### When to Use Multi-Head Attention
- Complex sequence modeling
- Transformer architectures
- Large-scale language models
- Tasks requiring multiple attention patterns

## 📊 Performance Considerations

### Memory Usage
- Self attention: O(n²) memory complexity
- Multi-head attention: O(n² × h) where h is number of heads

### Computational Complexity
- Both models: O(n²) time complexity
- Multi-head attention allows parallel computation

### Scaling Tips
- Use smaller embedding dimensions for memory constraints
- Reduce number of heads for faster training
- Consider sequence length impact on memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your attention mechanism implementation
4. Create corresponding documentation
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original attention paper: "Attention Is All You Need" by Vaswani et al.
- PyTorch community for the excellent deep learning framework
- The transformer architecture community for continuous improvements

---

**Happy Learning! 🚀**

For questions or issues, please open an issue on GitHub.
