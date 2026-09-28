import torch
from torch import nn


class MixtureOfExperts(nn.Module):
    def __init__(self,dim,num_experts=2):
        super().__init__(); self.experts=nn.ModuleList([nn.Sequential(nn.Linear(dim*4,dim*2),nn.ReLU(),nn.Linear(dim*2,dim)) for _ in range(num_experts)])
        self.router=nn.Linear(dim,num_experts)
    def forward(self,tokens,template):
        flat=tokens.flatten(1); rho=torch.softmax(self.router(template),-1)
        expert=torch.stack([e(flat) for e in self.experts],1)
        return (expert*rho.unsqueeze(-1)).sum(1),rho

