# Self Attention Mechanism

## 📖 Overview

Self Attention is the foundational attention mechanism where each position in a sequence attends to all positions in the same sequence, including itself. This allows the model to capture relationships between any two positions in the input sequence.

## 🧠 Theory

### What is Self Attention?

Self attention computes attention weights for each position by comparing it with every other position in the sequence. The key insight is that each position can directly access information from any other position through attention.

### Mathematical Formulation

The self attention mechanism follows this formula:

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

Where:
- **Q (Query)**: Linear projection of input to query space
- **K (Key)**: Linear projection of input to key space  
- **V (Value)**: Linear projection of input to value space
- **d_k**: Dimension of the key vectors
- **√d_k**: Scaling factor to prevent softmax saturation

### Step-by-Step Process

1. **Input Projection**: Transform input `x` into Q, K, V using linear layers
2. **Score Computation**: Calculate attention scores using `QK^T`
3. **Scaling**: Divide by `√d_k` to prevent large values
4. **Softmax**: Convert scores to probabilities
5. **Weighted Sum**: Apply attention weights to values
6. **Output Projection**: Transform the result with a final linear layer

## 🔧 Implementation Details

### Class Structure

```python
class SelfAttention(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        self.emd_dim = emd_dim
        self.q = nn.Linear(emd_dim, emd_dim)  # Query projection
        self.k = nn.Linear(emd_dim, emd_dim)  # Key projection
        self.v = nn.Linear(emd_dim, emd_dim)  # Value projection
        self.out_proj = nn.Linear(emd_dim, emd_dim)  # Output projection
```

### Forward Pass Breakdown

```python
def forward(self, x):
    # 1. Project input to Q, K, V
    Q = self.q(x)  # Shape: (batch, seq_len, embed_dim)
    K = self.k(x)  # Shape: (batch, seq_len, embed_dim)
    V = self.v(x)  # Shape: (batch, seq_len, embed_dim)
    
    # 2. Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
    
    # 3. Scale and apply softmax
    scores = scores / (self.emd_dim ** 0.5)
    attn_weight = F.softmax(scores, dim=-1)
    
    # 4. Apply attention to values
    context = torch.matmul(attn_weight, V)  # (batch, seq_len, embed_dim)
    
    # 5. Final projection
    output = self.out_proj(context)
    return output
```

## 📊 Complexity Analysis

### Time Complexity
- **O(n²)** where n is the sequence length
- Each position attends to all other positions

### Space Complexity  
- **O(n²)** for storing attention weights
- **O(n × d)** for storing Q, K, V projections

### Memory Usage
- Attention matrix: `batch_size × seq_len × seq_len`
- For long sequences, this can become memory-intensive

## 🎯 Use Cases

### When to Use Self Attention

✅ **Good for:**
- Short to medium sequence lengths (< 1000 tokens)
- Learning basic attention patterns
- Educational purposes
- Simple sequence modeling tasks
- Understanding attention fundamentals

❌ **Not ideal for:**
- Very long sequences (memory constraints)
- Tasks requiring multiple attention patterns
- Production transformer models (use multi-head instead)

## 🚀 Usage Examples

### Basic Usage

```python
import torch
from models.selfhead import SelfAttention

# Initialize model
embed_dim = 64
seq_len = 10
batch_size = 4

model = SelfAttention(emd_dim=embed_dim)

# Create input
x = torch.randn(batch_size, seq_len, embed_dim)

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")   # (4, 10, 64)
print(f"Output shape: {output.shape}")  # (4, 10, 64)
```

### Training with Main Interface

```bash
# Train self attention with default parameters
python main.py --model selfhead

# Custom parameters
python main.py --model selfhead --embed_dim 128 --epochs 200 --learning_rate 0.0001

# Interactive mode
python main.py --model selfhead --interactive
```

## 🔍 Understanding the Output

### Attention Weights Visualization

The attention weights matrix shows how much each position attends to every other position:

```
Position 0: [0.3, 0.1, 0.2, 0.4]  # Position 0 attends most to position 3
Position 1: [0.1, 0.5, 0.2, 0.2]  # Position 1 attends most to itself
Position 2: [0.2, 0.1, 0.6, 0.1]  # Position 2 attends most to itself
Position 3: [0.4, 0.2, 0.1, 0.3]  # Position 3 attends most to position 0
```

### Interpretation
- Higher values indicate stronger attention
- Diagonal values show self-attention
- Off-diagonal values show cross-position attention

## ⚡ Performance Tips

### Optimization Strategies

1. **Gradient Checkpointing**: For memory efficiency
2. **Mixed Precision**: Use `torch.float16` for faster training
3. **Sequence Length**: Keep sequences reasonably short
4. **Batch Size**: Adjust based on available memory

### Memory Management

```python
# For large sequences, consider gradient checkpointing
from torch.utils.checkpoint import checkpoint

# Use in forward pass
output = checkpoint(self.attention_forward, x)
```

## 🔬 Advanced Concepts

### Masked Self Attention

For tasks like language modeling, you might want to prevent positions from attending to future positions:

```python
def masked_attention(self, x, mask=None):
    # ... compute Q, K, V ...
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weight = F.softmax(scores, dim=-1)
    # ... rest of forward pass ...
```

### Positional Encoding

Self attention is position-agnostic. For sequence order, add positional encodings:

```python
# Add positional information
position_encoding = self.get_positional_encoding(seq_len, embed_dim)
x = x + position_encoding
```

## 🧪 Testing and Validation

### Unit Tests

```python
def test_self_attention():
    model = SelfAttention(emd_dim=32)
    x = torch.randn(2, 5, 32)
    
    # Test output shape
    output = model(x)
    assert output.shape == x.shape
    
    # Test attention weights sum to 1
    with torch.no_grad():
        Q = model.q(x)
        K = model.k(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (32 ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))
```

## 📚 Further Reading

- **"Attention Is All You Need"** - Vaswani et al. (2017)
- **"The Illustrated Transformer"** - Jay Alammar's blog post
- **"Attention and Augmented Recurrent Neural Networks"** - Olah & Carter

## 🔗 Related Models

- **[Multi-Head Attention](../multihead/README.md)** - Multiple attention heads in parallel
- **[Scaled Dot-Product Attention]** - The core attention mechanism
- **[Transformer Architecture]** - Complete transformer implementation

---

**Next Steps**: Try the [Multi-Head Attention](../multihead/README.md) for more advanced attention patterns! 