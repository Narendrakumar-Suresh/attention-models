import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAttention(nn.Module):
    def __init__(self, emd_dim,num_heads):
        super().__init__()
        assert emd_dim%num_heads==0

        self.emd_dim=emd_dim
        self.num_heads=num_heads
        self.head_dims=emd_dim//num_heads

        self.QKV=nn.Linear(emd_dim,3*emd_dim)
        self.out_proj=nn.Linear(emd_dim,emd_dim)

    def forward(self, x):
        B,T,D=x.size()
        QKV=self.QKV(x)
        Q,K,V=QKV.chunk(3,dim=-1)

        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dims).transpose(1, 2)
        Q=split_heads(Q)
        K=split_heads(K)
        V=split_heads(V)

        attn_scores=torch.matmul(Q,K.transpose(-2,-1))/(self.emd_dim**0.5)
        attn_weight=F.softmax(attn_scores,dim=-1)

        attn_output=torch.matmul(attn_weight,V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)


x = torch.randn(2, 5, 64)  # (batch=2, seq_len=5, embed_dim=64)
mha = MultiAttention(emd_dim=64, num_heads=8)
y = mha(x)
print(y.shape)
