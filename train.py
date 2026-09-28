"""Convenience entry point; QUICK_RUN remains controlled by config.py."""
from run_quick_test import Config,CachedLLMClient,run
if __name__ == "__main__":
    cfg=Config(); run(cfg,CachedLLMClient(cfg))

