import json, os, random, platform
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, allow_nan=True)


def environment_text():
    return "\n".join([
        f"python={platform.python_version()}", f"platform={platform.platform()}",
        f"torch={torch.__version__}", f"numpy={np.__version__}",
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
    ]) + "\n"


def clean(value, default="unknown"):
    if value is None or (isinstance(value, float) and np.isnan(value)): return default
    text = str(value).strip()
    return default if not text or text.lower() in {"nan", "not recorded", "none"} else text

