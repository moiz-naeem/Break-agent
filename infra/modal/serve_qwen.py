"""Serve Qwen3.8-27B on Modal with vLLM (OpenAI-compatible), with and without thinking.

One app, two endpoints, each on its own H200 that scales to zero when idle:
  qwen38          thinking on  (the model at its best)
  qwen38-nothink  thinking off (like-for-like baseline for a model trained on bare actions)

Deploy:            modal deploy infra/modal/serve_qwen.py
Fetch the weights: modal run infra/modal/serve_qwen.py::download   (CPU only, run once)

Both endpoints require "Authorization: Bearer $VLLM_API_KEY" (Modal secret `vllm-key`; a local
copy is in the git-ignored .env). Trained LoRA adapters go in the `railway-adapters` volume and
are loaded at run time through POST /v1/load_lora_adapter.
"""

import os
import subprocess

import modal

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
MINUTES = 60
MAX_MODEL_LEN = 131072  # a full-history Harbor episode is meant to stay under 100k tokens
GPU = os.environ.get("RAILWAY_GPU", "H200:1")  # read at deploy time, e.g. RAILWAY_GPU=A100-80GB

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
adapters = modal.Volume.from_name("railway-adapters", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache, "/adapters": adapters}

app = modal.App("railway-qwen")


@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache}, timeout=60 * MINUTES)
def download():
    """Pull the weights into the shared cache on a CPU container, so GPU time is not spent downloading."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, revision=REVISION)
    hf_cache.commit()
    print("downloaded to", path)


def _serve(served_name: str, thinking: bool) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", MODEL, "--revision", REVISION,
        "--served-model-name", served_name,
        "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enable-lora", "--max-lora-rank", "16",
    ]  # fmt: skip
    if not thinking:
        cmd += ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    return subprocess.Popen(cmd)


SERVER = dict(
    image=image,
    gpu=GPU,
    port=8000,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("vllm-key")],
    scaledown_window=5 * MINUTES,
    startup_timeout=30 * MINUTES,
    max_containers=1,
    target_concurrency=32,
    unauthenticated=True,  # Modal's own gate is off; vLLM checks the API key instead
)


@app.server(**SERVER)
class Think:
    @modal.enter()
    def start(self):
        self.process = _serve("qwen38", thinking=True)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.server(**SERVER)
class NoThink:
    @modal.enter()
    def start(self):
        self.process = _serve("qwen38-nothink", thinking=False)

    @modal.exit()
    def stop(self):
        self.process.terminate()
