import json, os, time, urllib.request
from pathlib import Path


class CachedLLMClient:
    """Optional OpenAI client. Every attempted request is followed by the configured pause."""
    def __init__(self, config, cache_name="llm_cache.json"):
        self.cfg = config; self.path = config.CACHE_DIR / cache_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.source_counts = {"cache": 0, "gpt": 0, "fallback": 0}

    def _key(self, kind, key): return f"{kind}:{key}"

    def generate(self, kind, key, prompt, fallback):
        ck = self._key(kind, key)
        if ck in self.cache:
            self.source_counts["cache"] += 1; return self.cache[ck]
        if self.cfg.USE_LLM:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key and self.cfg.API_KEY_FILE.exists():
                api_key = self.cfg.API_KEY_FILE.read_text(encoding="utf-8").strip()
            if api_key:
                for attempt in range(self.cfg.LLM_MAX_RETRIES):
                    try:
                        body = json.dumps({"model": self.cfg.LLM_MODEL, "input": prompt}).encode()
                        req = urllib.request.Request("https://api.openai.com/v1/responses", body,
                            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=60) as response:
                            data = json.load(response)
                        text = data["output"][0]["content"][0]["text"].strip()
                        self.source_counts["gpt"] += 1; self._save(ck, text); return text
                    except Exception as exc:
                        print(f"Warning: LLM attempt {attempt + 1} failed: {type(exc).__name__}")
                    finally:
                        # Mandatory throttle after every successful or failed request attempt.
                        time.sleep(max(5, self.cfg.LLM_SLEEP_SECONDS))
        self.source_counts["fallback"] += 1; self._save(ck, fallback); return fallback

    def _save(self, key, text):
        self.cache[key] = text
        self.path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

