"""Lifecycle management for the local Ollama daemon and the chat model.

`main.py` uses `OllamaService` as a context manager so a chat session is fully
self-service:

- on enter: start `ollama serve` if it isn't already up, check the model from
  `.env` is installed, then load it into RAM;
- on exit: unload the model (freeing its RAM) and shut the daemon down again if
  *we* were the ones who started it.

A daemon that was already running before RAGM4 started is left untouched.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from types import TracebackType

import httpx

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_REQUEST_TIMEOUT,
    require_ollama_model,
)

# How long to wait for a freshly spawned `ollama serve` to accept connections.
_DAEMON_STARTUP_TIMEOUT: float = 30.0
_DAEMON_POLL_INTERVAL: float = 0.5


class OllamaServiceError(RuntimeError):
    """Raised when the daemon or the requested model cannot be made ready."""


class OllamaService:
    """Start/stop the Ollama daemon and load/unload the configured model."""

    def __init__(self) -> None:
        self.model: str = require_ollama_model()
        self._process: subprocess.Popen[bytes] | None = None
        self._model_loaded: bool = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> OllamaService:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Ensure the daemon is running and the model is resident in RAM."""
        if self._is_daemon_up():
            print("[ollama] Daemon already running.")
        else:
            self._spawn_daemon()

        self._assert_model_installed()
        self._load_model()

    def stop(self) -> None:
        """Free the model's RAM and stop the daemon if we started it."""
        if self._model_loaded:
            self._unload_model()
        if self._process is not None:
            self._terminate_daemon()

    # -- daemon -------------------------------------------------------------

    def _is_daemon_up(self) -> bool:
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK

    def _spawn_daemon(self) -> None:
        if shutil.which("ollama") is None:
            raise OllamaServiceError(
                "The `ollama` command was not found on your PATH.\n"
                "Install it from https://ollama.com, then pull a model with\n"
                "  ollama pull qwen2.5:14b\n"
                "and put the tag in your .env file as OLLAMA_MODEL=…"
            )

        print("[ollama] Starting daemon (`ollama serve`) …")
        self._process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + _DAEMON_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._is_daemon_up():
                print("[ollama] Daemon ready.")
                return
            if self._process.poll() is not None:
                self._process = None
                raise OllamaServiceError(
                    "`ollama serve` exited immediately. Another daemon may "
                    f"already own {OLLAMA_BASE_URL}; check with `ollama ls`."
                )
            time.sleep(_DAEMON_POLL_INTERVAL)

        self._terminate_daemon()
        raise OllamaServiceError(
            f"Ollama daemon did not become ready within "
            f"{_DAEMON_STARTUP_TIMEOUT:.0f}s at {OLLAMA_BASE_URL}."
        )

    def _terminate_daemon(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return

        print("[ollama] Stopping the daemon we started …")
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    # -- model --------------------------------------------------------------

    def _installed_models(self) -> list[str]:
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as err:  # pragma: no cover - network failure path
            raise OllamaServiceError(
                f"Could not query the Ollama daemon at {OLLAMA_BASE_URL}: {err}"
            ) from err
        return [model["name"] for model in response.json().get("models", [])]

    def _assert_model_installed(self) -> None:
        installed = self._installed_models()
        # `ollama ls` shows "name:tag"; a bare "name" means the ":latest" tag.
        candidates = {self.model, f"{self.model}:latest"}
        if candidates.isdisjoint(installed):
            available = "\n".join(f"    - {name}" for name in installed) or "    (none)"
            raise OllamaServiceError(
                f"Model '{self.model}' is not installed in Ollama.\n"
                f"Pull it first:\n"
                f"    ollama pull {self.model}\n"
                f"Models currently available:\n{available}\n"
                "Then set OLLAMA_MODEL in your .env file to one of those tags."
            )

    def _load_model(self) -> None:
        """Warm the model up so the first question isn't slowed by the load."""
        print(f"[ollama] Loading model '{self.model}' into memory …")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
                timeout=OLLAMA_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as err:
            raise OllamaServiceError(
                f"Failed to load model '{self.model}': {err}"
            ) from err

        self._model_loaded = True
        print(f"[ollama] Model '{self.model}' is served and ready.")

    def _unload_model(self) -> None:
        """Ask Ollama to evict the model immediately (`keep_alive: 0`)."""
        self._model_loaded = False
        print(f"[ollama] Unloading '{self.model}' to free memory …")
        try:
            httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=30.0,
            ).raise_for_status()
        except httpx.HTTPError as err:
            print(f"[ollama] Warning: could not unload the model ({err}).")
            return
        print("[ollama] Memory freed.")
