import torch
from torch import nn


class PSTAL(nn.Module):
    def __init__(self,input_dims,latent_dim):
        super().__init__(); self.proj=nn.ModuleList([nn.Sequential(nn.Linear(d,latent_dim),nn.LayerNorm(latent_dim)) for d in input_dims])
        self.attn=nn.Linear(latent_dim,1)
    def forward(self,features):
        z=torch.stack([p(x) for p,x in zip(self.proj,features)],1)
        alpha=torch.softmax(self.attn(z).squeeze(-1),1); template=(z*alpha.unsqueeze(-1)).sum(1)
        loss=((z-template.unsqueeze(1))**2).mean()
        return z,template,alpha,loss

