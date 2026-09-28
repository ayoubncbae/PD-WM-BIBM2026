import torch
from torch import nn


class StructuredEnergyTransition(nn.Module):
    def __init__(self,dim,text_dim):
        super().__init__(); self.text_proj=nn.Linear(text_dim,dim)
        self.interact=nn.Sequential(nn.Linear(dim*7,dim*2),nn.ReLU(),nn.Linear(dim*2,dim),nn.ReLU())
        self.energy=nn.Linear(dim,1); self.use=nn.Linear(dim,2); self.dose=nn.Linear(dim,4); self.intensity=nn.Linear(dim,4)
    def forward(self,state,medicine,pre,effect,applicability,severity,candidates):
        action=[self.text_proj(x) for x in (medicine,pre,effect,applicability)]
        candidates=self.text_proj(candidates)
        b,c=candidates.shape[:2]; parts=[state,*action,severity]
        base=torch.cat(parts,-1).unsqueeze(1).expand(-1,c,-1)
        h=self.interact(torch.cat([base,candidates],-1)); energies=self.energy(h).squeeze(-1)
        probs=torch.softmax(-energies,-1); context=(h*probs.unsqueeze(-1)).sum(1)
        return probs,context,self.use(context),self.dose(context),self.intensity(context)
