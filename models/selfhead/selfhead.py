import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        self.emd_dim=emd_dim
        self.q=nn.Linear(emd_dim,emd_dim)
        self.k=nn.Linear(emd_dim,emd_dim)
        self.v=nn.Linear(emd_dim,emd_dim)
        self.out_proj=nn.Linear(emd_dim,emd_dim)

    def forward(self, x):

        Q=self.q(x)
        K=self.k(x)
        V=self.v(x)
        # Compute attention matmul
        scores=torch.matmul(Q,K.transpose(-2,-1))/(self.emd_dim**0.5)
        attn_weight=F.softmax(scores,dim=-1)
        context=torch.matmul(attn_weight,V)

        output=self.out_proj(context)

        return output


x=torch.randn(2,5,32)
att=SelfAttention(emd_dim=32)
y=att(x)
print(y.shape)
