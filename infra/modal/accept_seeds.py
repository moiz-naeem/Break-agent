"""Build the accepted dev seed lists (T06) on Modal CPU containers, one per preset, in parallel.

The seed filter is plain Python (greedy, the reference and the exploits on each candidate scenario),
so it needs many CPU cores, not a GPU.

Run:  modal run infra/modal/accept_seeds.py                       (T1 and T2, 20 seeds each)
      modal run infra/modal/accept_seeds.py --presets T2 --count 30

The code sent to Modal is the committed HEAD (`git archive`), with LF line endings. The list's
`generator_version` is a hash of the generator source bytes, so a CRLF checkout gives another hash.
"""

import io
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

CORES = 32

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "httpx>=0.28.1", "matplotlib>=3.11.2", "mcp>=2.3.0", "numpy>=2.4.6", "pydantic>=2.14.3", "pyyaml>=6.0.3"
)
if modal.is_local():  # inside the container this file sits at /root, with no repo around it
    REPO = Path(__file__).resolve().parents[2]
    source = Path(tempfile.mkdtemp(prefix="railway-src-"))
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "railway", "seeds"], capture_output=True)
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(source)
    image = image.add_local_dir(source, "/src")

app = modal.App("railway-accept-seeds", image=image)


@app.function(cpu=CORES, memory=32768, timeout=4 * 3600)
def accept(preset: str, set_name: str, count: int) -> str:
    import os
    import sys
    import time

    os.chdir("/src")
    sys.path.insert(0, "/src")
    from railway.ref.seed_filter import accepted_seeds

    start = time.monotonic()
    accepted_seeds(preset, set_name, count, output_dir="/tmp/out", workers=CORES)
    print(f"{preset}-{set_name}: {count} accepted in {time.monotonic() - start:.0f} s")
    return Path(f"/tmp/out/{preset}-{set_name}.json").read_text(encoding="utf-8")


@app.local_entrypoint()
def main(presets: str = "T1,T2", set_name: str = "dev", count: int = 20):
    names = presets.split(",")
    for preset, text in zip(names, accept.starmap([(p, set_name, count) for p in names]), strict=True):
        path = REPO / "seeds" / f"{preset}-{set_name}.json"
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote", path)
