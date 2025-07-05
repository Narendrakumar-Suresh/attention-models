# Multi-Head Attention Mechanism

## 📖 Overview

Multi-Head Attention is an advanced attention mechanism that uses multiple attention heads in parallel, allowing the model to attend to different types of information simultaneously. This is the attention mechanism used in the original Transformer architecture and is more powerful than single-head attention.

## 🧠 Theory

### What is Multi-Head Attention?

Multi-head attention runs several attention mechanisms in parallel, each with its own set of learned parameters. This allows the model to focus on different aspects of the input simultaneously - for example, one head might focus on local relationships while another focuses on global patterns.

### Why Multiple Heads?

1. **Diverse Attention Patterns**: Different heads can learn different types of relationships
2. **Enhanced Representation**: Multiple perspectives lead to richer representations
3. **Parallel Computation**: Heads can be computed in parallel for efficiency
4. **Robustness**: Multiple heads provide redundancy and stability

### Mathematical Formulation

Multi-head attention is defined as:

```
MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h)W^O
```

Where each head is:

```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

And:

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

## 🔧 Implementation Details

### Class Structure

```python
class MultiAttention(nn.Module):
    def __init__(self, emd_dim, num_heads):
        super().__init__()
        assert emd_dim % num_heads == 0  # Must be divisible
        
        self.emd_dim = emd_dim
        self.num_heads = num_heads
        self.head_dims = emd_dim // num_heads  # Dimension per head
        
        # Single linear layer for QKV projection
        self.QKV = nn.Linear(emd_dim, 3 * emd_dim)
        self.out_proj = nn.Linear(emd_dim, emd_dim)
```

### Key Design Decisions

1. **Single QKV Projection**: More efficient than separate Q, K, V projections
2. **Head Dimension Calculation**: `head_dims = emd_dim // num_heads`
3. **Divisibility Check**: Ensures clean splitting into heads

### Forward Pass Breakdown

```python
def forward(self, x):
    B, T, D = x.size()  # batch, seq_len, embed_dim
    
    # 1. Project to QKV space
    QKV = self.QKV(x)  # Shape: (batch, seq_len, 3 * embed_dim)
    Q, K, V = QKV.chunk(3, dim=-1)  # Split into Q, K, V
    
    # 2. Reshape for multi-head attention
    def split_heads(tensor):
        return tensor.view(B, T, self.num_heads, self.head_dims).transpose(1, 2)
    
    Q = split_heads(Q)  # Shape: (batch, num_heads, seq_len, head_dims)
    K = split_heads(K)  # Shape: (batch, num_heads, seq_len, head_dims)
    V = split_heads(V)  # Shape: (batch, num_heads, seq_len, head_dims)
    
    # 3. Compute attention for each head
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.emd_dim ** 0.5)
    attn_weight = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weight, V)
    
    # 4. Concatenate heads and project
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
    return self.out_proj(attn_output)
```

## 📊 Complexity Analysis

### Time Complexity
- **O(n² × h)** where n is sequence length, h is number of heads
- Each head computes attention independently
- Parallel computation possible

### Space Complexity
- **O(n² × h)** for storing attention weights across all heads
- **O(n × d × h)** for storing Q, K, V across all heads

### Memory Usage
- Attention matrices: `batch_size × num_heads × seq_len × seq_len`
- More memory-intensive than single-head attention

## 🎯 Use Cases

### When to Use Multi-Head Attention

✅ **Excellent for:**
- Transformer architectures
- Large language models
- Complex sequence modeling
- Tasks requiring multiple attention patterns
- Production systems
- Long-range dependencies

✅ **Good for:**
- Medium to long sequences
- Tasks with diverse information types
- When you need robust attention patterns

❌ **Consider alternatives for:**
- Very short sequences (< 10 tokens)
- Memory-constrained environments
- Simple tasks where single-head suffices

## 🚀 Usage Examples

### Basic Usage

```python
import torch
from models.multihead import MultiAttention

# Initialize model
embed_dim = 64
num_heads = 8
seq_len = 10
batch_size = 4

model = MultiAttention(emd_dim=embed_dim, num_heads=num_heads)

# Create input
x = torch.randn(batch_size, seq_len, embed_dim)

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")   # (4, 10, 64)
print(f"Output shape: {output.shape}")  # (4, 10, 64)
```

### Training with Main Interface

```bash
# Train multi-head attention with default parameters
python main.py --model multihead

# Custom parameters
python main.py --model multihead --embed_dim 256 --num_heads 16 --epochs 300

# Interactive mode
python main.py --model multihead --interactive
```

### Parameter Guidelines

| Embed Dim | Recommended Heads | Use Case |
|-----------|------------------|----------|
| 64 | 8 | Small models, learning |
| 128 | 8-16 | Medium models |
| 256 | 16-32 | Large models |
| 512 | 32-64 | Very large models |
| 1024+ | 64+ | Massive models |

## 🔍 Understanding Multi-Head Output

### Head Specialization

Different heads often learn to attend to different patterns:

```
Head 0: [0.8, 0.1, 0.1]  # Attends to first position
Head 1: [0.1, 0.8, 0.1]  # Attends to second position  
Head 2: [0.3, 0.3, 0.4]  # Distributed attention
Head 3: [0.5, 0.3, 0.2]  # Local attention pattern
```

### Visualization Example

```python
def visualize_attention_heads(model, x):
    # Get attention weights for each head
    B, T, D = x.size()
    QKV = model.QKV(x)
    Q, K, V = QKV.chunk(3, dim=-1)
    
    # Reshape for heads
    Q = Q.view(B, T, model.num_heads, model.head_dims).transpose(1, 2)
    K = K.view(B, T, model.num_heads, model.head_dims).transpose(1, 2)
    
    # Compute attention weights
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (model.emd_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    
    return attn_weights  # Shape: (batch, num_heads, seq_len, seq_len)
```

## ⚡ Performance Optimization

### Memory Efficiency

```python
# Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self._forward_impl, x)

# Mixed precision training
from torch.cuda.amp import autocast

with autocast():
    output = model(x)
```

### Parallel Processing

```python
# Multi-GPU training
model = nn.DataParallel(model)

# Distributed training
model = nn.parallel.DistributedDataParallel(model)
```

### Head Pruning

For inference, you can prune less important heads:

```python
def prune_heads(self, heads_to_prune):
    """Remove specific attention heads"""
    # Implementation for head pruning
    pass
```

## 🔬 Advanced Concepts

### Relative Positional Encoding

For better position awareness:

```python
def relative_positional_encoding(self, seq_len, head_dim):
    """Add relative positional information"""
    pos_encoding = torch.zeros(seq_len, head_dim)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    
    div_term = torch.exp(torch.arange(0, head_dim, 2).float() * 
                        -(math.log(10000.0) / head_dim))
    
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding
```

### Sparse Attention

For very long sequences:

```python
def sparse_attention(self, Q, K, V, sparsity_factor=4):
    """Implement sparse attention for efficiency"""
    # Only attend to every k-th position
    K_sparse = K[:, :, ::sparsity_factor, :]
    V_sparse = V[:, :, ::sparsity_factor, :]
    
    scores = torch.matmul(Q, K_sparse.transpose(-2, -1))
    attn_weights = F.softmax(scores, dim=-1)
    
    return torch.matmul(attn_weights, V_sparse)
```

## 🧪 Testing and Validation

### Comprehensive Tests

```python
def test_multihead_attention():
    model = MultiAttention(emd_dim=64, num_heads=8)
    x = torch.randn(2, 10, 64)
    
    # Test output shape
    output = model(x)
    assert output.shape == x.shape
    
    # Test head dimension constraint
    try:
        bad_model = MultiAttention(emd_dim=63, num_heads=8)  # Should fail
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass
    
    # Test attention weights normalization
    with torch.no_grad():
        QKV = model.QKV(x)
        Q, K, V = QKV.chunk(3, dim=-1)
        Q = Q.view(2, 10, 8, 8).transpose(1, 2)
        K = K.view(2, 10, 8, 8).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))
```

## 📚 Further Reading

- **"Attention Is All You Need"** - Vaswani et al. (2017)
- **"Analyzing Multi-Head Self-Attention"** - Michel et al. (2019)
- **"Are Sixteen Heads Really Better than One?"** - Michel et al. (2019)
- **"The Illustrated Transformer"** - Jay Alammar's blog post

## 🔗 Related Models

- **[Self Attention](../selfhead/README.md)** - Single-head attention mechanism
- **[Transformer Architecture]** - Complete transformer implementation
- **[BERT/GPT]** - Large language models using multi-head attention

## 🎯 Best Practices

### Head Configuration

1. **Number of Heads**: Usually 8-64, must divide embed_dim evenly
2. **Head Dimension**: Typically 32-128 dimensions per head
3. **Scaling**: More heads for larger models

### Training Tips

1. **Learning Rate**: Start with 1e-4 to 1e-3
2. **Warmup**: Use learning rate warmup for stable training
3. **Regularization**: Dropout on attention weights (0.1-0.2)
4. **Initialization**: Use proper weight initialization

### Debugging

```python
# Check attention patterns
def debug_attention(model, x):
    with torch.no_grad():
        output = model(x)
        # Analyze attention weights
        # Check for dead heads
        # Verify gradient flow
    return output
```

---

**Next Steps**: Explore the complete [Transformer Architecture](../README.md) or try [Self Attention](../selfhead/README.md) for comparison! 