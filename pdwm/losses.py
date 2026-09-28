import torch
import torch.nn.functional as F


def masked_ce(logits,target):
    mask=target>=0
    return F.cross_entropy(logits[mask],target[mask]) if mask.any() else logits.sum()*0


def joint_loss(output,batch,cfg):
    sev=F.nll_loss(torch.log(output["severity_probs"]),batch["severity"])
    use=masked_ce(output["use_logits"],batch["use"])
    state=masked_ce(torch.log(output["management_probs"]),batch["management"])
    dose=masked_ce(output["dose_logits"],batch["dose"]); intensity=masked_ce(output["intensity_logits"],batch["intensity"])
    slot=(dose+intensity)/2
    total=sev+cfg.LAMBDA_P*output["pstal_loss"]+cfg.LAMBDA_W*(cfg.LAMBDA_USE*use+cfg.LAMBDA_STATE*state+cfg.LAMBDA_SLOT*slot)
    return total,{"severity":sev.item(),"pstal":output["pstal_loss"].item(),"use":use.item(),"state":state.item(),"slot":slot.item(),"total":total.item()}

