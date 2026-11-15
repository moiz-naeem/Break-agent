"""A vLLM OpenAI-compatible server as a subprocess on its own GPU, with runtime LoRA loading.

Used on Modal next to the trainer (same container, another GPU), so a freshly saved adapter is on the
local disk and loads at once with POST /v1/load_lora_adapter. Flags match infra/modal/serve_qwen.py.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx

from railway.train.policy import LORA_RANK, MODEL, REVISION


class LocalVLLM:
    def __init__(
        self,
        model: str = MODEL,
        revision: str = REVISION,
        *,
        served_name: str = "qwen38",
        port: int = 8001,
        gpus: str = "0",
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
        max_loras: int = 2,
        log_path: str | Path = "/tmp/vllm.log",
        extra_args: tuple[str, ...] = (),
    ):
        self.model, self.revision, self.served_name, self.port = model, revision, served_name, port
        self.gpus = gpus
        tp = len(gpus.split(","))
        self.cmd = [
            "vllm", "serve", model, "--revision", revision,
            "--served-model-name", served_name,
            "--host", "127.0.0.1", "--port", str(port),
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--enable-lora", "--max-lora-rank", str(LORA_RANK), "--max-loras", str(max_loras),
            "--tensor-parallel-size", str(tp),
            *extra_args,
        ]  # fmt: skip
        self.log_path = Path(log_path)
        self.process: subprocess.Popen | None = None
        self.loaded: list[str] = []

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def base_url(self) -> str:
        return self.base + "/v1"

    def start(self) -> LocalVLLM:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpus, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        self._log = self.log_path.open("w")
        self.process = subprocess.Popen(self.cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT)
        return self

    def wait(self, timeout_s: float = 1800) -> float:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.process.returncode}:\n{self.tail()}")
            try:
                if httpx.get(self.base + "/health", timeout=5).status_code == 200:
                    return time.monotonic() - start
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not healthy after {timeout_s} s:\n{self.tail()}")

    def tail(self, n: int = 60) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def load_lora(self, name: str, path: str) -> str:
        r = httpx.post(self.base + "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path}, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"load_lora_adapter {name}: HTTP {r.status_code} {r.text[:500]}")
        self.loaded.append(name)
        return r.text

    def unload_lora(self, name: str) -> None:
        r = httpx.post(self.base + "/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"unload_lora_adapter {name}: HTTP {r.status_code} {r.text[:500]}")
        self.loaded.remove(name)

    def swap_lora(self, name: str, path: str) -> None:
        """Load the new adapter under a new name, then drop older ones (names are never reused, so no stale cache)."""
        self.load_lora(name, path)
        for old in [n for n in self.loaded if n != name]:
            self.unload_lora(old)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                self.process.kill()
