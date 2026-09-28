import torch
from torch import nn
from .encoders import IJEPAVisionEncoder,GatedBagAggregator
from .pstal import PSTAL
from .moe import MixtureOfExperts
from .ordinal import OrdinalHead
from .energy_transition import StructuredEnergyTransition


class PDWorldModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.VISUAL_DIM; l=cfg.LATENT_DIM
        self.visual=IJEPAVisionEncoder(d,cfg.IJEPA_BACKEND,cfg.IJEPA_CHECKPOINT); self.bag=GatedBagAggregator(d)
        self.pstal=PSTAL([d,d,cfg.TEXT_DIM],l); self.moe=MixtureOfExperts(l,cfg.NUM_EXPERTS); self.ordinal=OrdinalHead(l)
        self.severity_emb=nn.Embedding(3,l); self.transition=StructuredEnergyTransition(l,cfg.TEXT_DIM)
    def image_bag(self,x,mask):
        b,n,c,h,w=x.shape; feats=self.visual(x.reshape(b*n,c,h,w)).reshape(b,n,-1); return self.bag(feats,mask)[0]
    def forward(self,batch):
        mri=self.image_bag(batch["mri"],batch["mri_mask"]); dat=self.image_bag(batch["dat"],batch["dat_mask"])
        z,t,alpha,pstal=self.pstal([mri,dat,batch["narrative"]]); tokens=torch.cat([z,t.unsqueeze(1)],1)
        state,rho=self.moe(tokens,t); sev_probs=self.ordinal(state)
        sev_soft=sev_probs@self.severity_emb.weight
        mgmt,ctx,use,dose,intensity=self.transition(state,batch["medicine"],batch["pre"],batch["effect"],batch["applicability"],sev_soft,batch["candidates"])
        return {"severity_probs":sev_probs,"management_probs":mgmt,"use_logits":use,"dose_logits":dose,"intensity_logits":intensity,
                "pstal_loss":pstal,"router":rho,"modality_attention":alpha}

