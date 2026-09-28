from config import Config
from pdwm.llm_client import CachedLLMClient
from pdwm.pipeline import run


if __name__ == "__main__":
    cfg=Config(); client=CachedLLMClient(cfg)
    result=run(cfg,client)
    print("="*50)
    print("PD-WM QUICK PIPELINE TEST COMPLETED")
    print(f"Subjects used: {result['subjects']}")
    print(f"Train/Val/Test: {result['splits'].get('train',0)}/{result['splits'].get('val',0)}/{result['splits'].get('test',0)}")
    print("Visual encoder: PLACEHOLDER")
    print("Text encoder: PLACEHOLDER")
    print(f"LLM narratives: {result['llm']}")
    print("Training completed: YES")
    print("Inference completed: YES")
    print(f"Severity accuracy: {result['severity']['accuracy']}")
    print(f"Management Top-1: {result['management']['structured_state_top1_accuracy']}")
    print(f"Outputs: {cfg.OUTPUT_DIR}")
    print("="*50)

