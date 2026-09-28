import hashlib
import numpy as np
import torch
from torch import nn


class IJEPAVisionEncoder(nn.Module):
    """I-JEPA-compatible image interface; the default backend is a small CNN placeholder."""
    def __init__(self, output_dim=64, backend="placeholder", checkpoint=None):
        super().__init__(); self.backend=backend
        if backend != "placeholder":
            raise NotImplementedError("Connect the licensed I-JEPA loader here and set IJEPA_BACKEND after placing its checkpoint.")
        self.net=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                               nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,output_dim))
    def forward(self,x): return self.net(x)


class GatedBagAggregator(nn.Module):
    def __init__(self, dim): super().__init__(); self.gate=nn.Linear(dim,1)
    def forward(self,x,mask):
        logits=self.gate(x).squeeze(-1).masked_fill(mask<=0,-1e4)
        weights=torch.softmax(logits,1)*mask
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        return (x*weights.unsqueeze(-1)).sum(1),weights


class MedCPTTextEncoder:
    """MedCPT-compatible deterministic hashing placeholder with no fitted vocabulary."""
    def __init__(self, output_dim=128, backend="placeholder", model_path=None):
        self.output_dim=output_dim; self.backend=backend
        if backend != "placeholder":
            raise NotImplementedError("Connect a local transformers MedCPT loader here after setting MEDCPT_MODEL_PATH.")
    def encode(self,texts):
        out=np.zeros((len(texts),self.output_dim),dtype="float32")
        for i,text in enumerate(texts):
            for token in str(text).lower().split():
                digest=hashlib.sha256(token.encode()).digest(); j=int.from_bytes(digest[:4],"little")%self.output_dim
                out[i,j] += 1 if digest[4]%2 else -1
            norm=np.linalg.norm(out[i]); out[i]/=norm if norm else 1
        return torch.from_numpy(out)
