import torch
from torch import nn
import torch.nn.functional as F


class OrdinalHead(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.score=nn.Linear(dim,1); self.theta1=nn.Parameter(torch.tensor(-0.5)); self.delta=nn.Parameter(torch.tensor(0.5))
    def forward(self,x):
        g=self.score(x).squeeze(-1); t2=self.theta1+F.softplus(self.delta)
        q1=torch.sigmoid(g-self.theta1); q2=torch.sigmoid(g-t2)
        return torch.stack([1-q1,q1-q2,q2],-1).clamp_min(1e-7)

