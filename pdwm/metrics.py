import warnings
import numpy as np
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,
    matthews_corrcoef,cohen_kappa_score,roc_auc_score,average_precision_score,confusion_matrix)
from sklearn.preprocessing import label_binarize


def safe_metric(name,fn):
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); value=fn()
        if not np.isfinite(value): raise ValueError
        return float(value)
    except Exception:
        print(f"{name}: Metric undefined in small sanity subset."); return float("nan")


def severity_metrics(y,p,probs):
    y=np.asarray(y); p=np.asarray(p); probs=np.asarray(probs)
    cm=confusion_matrix(y,p,labels=[0,1,2]); specs=[]
    for i in range(3):
        tn=cm.sum()-cm[i].sum()-cm[:,i].sum()+cm[i,i]; fp=cm[:,i].sum()-cm[i,i]
        if tn+fp: specs.append(tn/(tn+fp))
    onehot=label_binarize(y,classes=[0,1,2]); ece=0
    conf=probs.max(1); correct=(p==y).astype(float)
    for lo in np.linspace(0,1,11)[:-1]:
        mask=(conf>=lo)&(conf<lo+.1)
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return {
        "accuracy":safe_metric("Accuracy",lambda:accuracy_score(y,p)),
        "balanced_accuracy":safe_metric("Balanced Accuracy",lambda:balanced_accuracy_score(y,p)),
        "macro_precision":safe_metric("Macro Precision",lambda:precision_score(y,p,average="macro",zero_division=0)),
        "macro_recall":safe_metric("Macro Recall",lambda:recall_score(y,p,average="macro",zero_division=0)),
        "macro_f1":safe_metric("Macro F1",lambda:f1_score(y,p,average="macro",zero_division=0)),
        "weighted_f1":safe_metric("Weighted F1",lambda:f1_score(y,p,average="weighted",zero_division=0)),
        "mcc":safe_metric("MCC",lambda:matthews_corrcoef(y,p)),"cohen_kappa":safe_metric("Kappa",lambda:cohen_kappa_score(y,p)),
        "macro_specificity":float(np.mean(specs)) if specs else float("nan"),
        "roc_auc":safe_metric("ROC-AUC",lambda:roc_auc_score(onehot,probs,average="macro",multi_class="ovr")),
        "pr_auc":safe_metric("PR-AUC",lambda:average_precision_score(onehot,probs,average="macro")),
        "ece":float(ece),"brier_score":float(np.mean(np.sum((probs-onehot)**2,axis=1)))
    }


def classification_slot(y,p,prefix):
    mask=np.asarray(y)>=0
    if not mask.any(): return {f"{prefix}_accuracy":float("nan"),f"{prefix}_f1":float("nan")}
    ya=np.asarray(y)[mask]; pa=np.asarray(p)[mask]
    return {f"{prefix}_accuracy":float(accuracy_score(ya,pa)),f"{prefix}_f1":float(f1_score(ya,pa,average="macro",zero_division=0))}

