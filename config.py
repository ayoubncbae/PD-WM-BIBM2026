from dataclasses import dataclass, asdict
from pathlib import Path
import torch


@dataclass
class Config:
    PROJECT_DIR: Path = Path(__file__).resolve().parent
    DATA_ROOT: Path = PROJECT_DIR.parent / "OpenNeuro_PD_data"
    PATIENT_CSV: Path = PROJECT_DIR.parent / "neuroplan_pd_patient_master_with_severity.csv"
    API_KEY_FILE: Path = PROJECT_DIR.parent / "api_key.txt"
    OUTPUT_DIR: Path = PROJECT_DIR / "outputs"
    CACHE_DIR: Path = PROJECT_DIR / "cache"
    QUICK_RUN: bool = True
    MAX_SUBJECTS: int = 10
    SEED: int = 42
    EPOCHS: int = 3
    BATCH_SIZE: int = 4
    LR: float = 1e-3
    IMAGE_SIZE: int = 96
    BAG_SIZE: int = 4
    VISUAL_DIM: int = 64
    TEXT_DIM: int = 128
    LATENT_DIM: int = 64
    NUM_EXPERTS: int = 2
    LAMBDA_P: float = 0.1
    LAMBDA_W: float = 1.0
    LAMBDA_USE: float = 1.0
    LAMBDA_STATE: float = 1.0
    LAMBDA_SLOT: float = 1.0
    USE_LLM: bool = False
    LLM_MODEL: str = "gpt-4.1-mini"
    LLM_SLEEP_SECONDS: int = 5
    LLM_MAX_RETRIES: int = 2
    IJEPA_BACKEND: str = "placeholder"
    MEDCPT_BACKEND: str = "placeholder"
    IJEPA_CHECKPOINT: str = "weights/ijepa/PUT_IJEPA_WEIGHTS_HERE"
    MEDCPT_MODEL_PATH: str = "weights/medcpt/PUT_MEDCPT_MODEL_HERE"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def serializable(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

