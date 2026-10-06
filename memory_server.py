"""
memory_server.py - the fleet's long-term memory (Hindsight), hosted by DREAM.

DREAM is the Distributed Runtime for Ethereal Autonomous Memories, so the
fleet's memory lives with her: run.bat starts this when
Config.DREAM.Memory.Hindsight is true, and TINA (wherever she runs) connects
to it over the network.

One server, one bank per agent: "tina" (TINA's projects and decisions) and
"dream" (DREAM's life with you). Fully local: the language model it uses to
pull facts out of memories is Ollama's, embeddings and reranking run on the
CPU with ONNX (no PyTorch), and the database is Hindsight's embedded Postgres
(in its default place).

NOTE(move): this is meant to run on the other PC, next to TINA, where it can
use a bigger model and the GPU. Moving it = run this there, then point
TINA's HindsightUrl and DREAM's Memory.HindsightUrl at that PC.

    hindsight-venv\\Scripts\\python.exe hindsight_server.py      (run.bat does this)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def settings():
    with open(ROOT / "config.json", encoding="utf-8") as f:
        h = json.load(f)["Config"]["DREAM"].get("Memory", {})
    return {
        "HINDSIGHT_API_HOST": "0.0.0.0",
        "HINDSIGHT_API_PORT": str(h.get("Port", 8888)),
        "HINDSIGHT_API_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_LLM_MODEL": h.get("LlmModel", "qwen3:4b"),
        "HINDSIGHT_API_LLM_BASE_URL": h.get("OllamaUrl", "http://localhost:11434").rstrip("/") + "/v1",
        "HINDSIGHT_API_LLM_API_KEY": "ollama",
        "HINDSIGHT_API_LLM_REASONING_EFFORT": "none",          # no thinking blocks: fact extraction, not puzzles
        "HINDSIGHT_API_LLM_STRICT_SCHEMA": "true",              # small local models need their JSON enforced
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": h.get("EmbeddingsProvider", "onnx"),
        "HINDSIGHT_API_RERANKER_PROVIDER": h.get("RerankerProvider", "flashrank"),
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU": "true",
        "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU": "true",
    }


def main():
    env = dict(os.environ, **settings())
    exe = Path(sys.executable).with_name("hindsight-api.exe" if os.name == "nt" else "hindsight-api")
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "hindsight_api"]
    print(f"Hindsight memory on http://127.0.0.1:{env['HINDSIGHT_API_PORT']}/  (model {env['HINDSIGHT_API_LLM_MODEL']} via Ollama, CPU embeddings)")
    return subprocess.call(cmd, env=env, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
