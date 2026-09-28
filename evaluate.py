"""Print saved quick-run evaluation metrics without retraining."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent/"outputs"
for name in ("severity_metrics.json","management_metrics.json"):
    print(name); print(json.dumps(json.loads((root/name).read_text()),indent=2))

