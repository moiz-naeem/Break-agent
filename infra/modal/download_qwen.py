"""Pull the Qwen3.8-27B weights into the shared Modal volume on a CPU container (no GPU needed).

Run once:  modal run infra/modal/download_qwen.py
"""

import modal

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub", "hf_xet")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("railway-qwen-download")


@app.function(image=image, volumes={"/root/.cache/huggingface": hf_cache}, timeout=3600, cpu=4, memory=8192)
def download():
    import os

    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, revision=REVISION)
    hf_cache.commit()
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
    print(f"downloaded to {path}: {size / 1e9:.1f} GB")
