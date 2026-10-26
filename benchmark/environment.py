
"""
Environment metadata collector for LLM Serving Benchmarks.

Collects:
1. Git revision and working-tree status
2. Operating system and client Python environment
3. GPU hardware and NVIDIA driver
4. Package versions from the separate vLLM environment
5. Running vLLM server process and its startup arguments

Linux/WSL2 is required for live process inspection.

Usage:
    python benchmark/environment.py
"""

import json
import platform
import re
import shlex
import subprocess
import sys

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVER_PYTHON = (
    Path.home()
    / "venvs"
    / "llm-serving-vllm"
    / "bin"
    / "python"
)

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000


# ------------------------------------------------------------
# General utilities
# ------------------------------------------------------------

def run_command(command: list[str]) -> str | None:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as exc:
        print(
            f"[Environment] Command failed: {exc}",
            file=sys.stderr,
        )
        return None


def package_version(package: str) -> str | None:
    """Get a package version from the active client environment."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


# ------------------------------------------------------------
# Installed vLLM environment
# ------------------------------------------------------------

def collect_server_versions() -> dict:
    """
    Query the separate vLLM virtual environment.

    These are installed package versions, not proof that
    the running server was launched with these versions.
    """
    if not SERVER_PYTHON.is_file():
        raise FileNotFoundError(
            f"Server Python not found: {SERVER_PYTHON}"
        )

    script = """
import json
from importlib.metadata import version
import torch

print(json.dumps({
    "vllm_version": version("vllm"),
    "torch_version": version("torch"),
    "transformers_version": version("transformers"),
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}))
"""

    try:
        process = subprocess.run(
            [str(SERVER_PYTHON), "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )

        lines = [
            line.strip()
            for line in process.stdout.splitlines()
            if line.strip()
        ]

        if not lines:
            raise ValueError("Empty server version output")

        result = json.loads(lines[-1])

        required = {
            "vllm_version",
            "torch_version",
            "transformers_version",
            "cuda_runtime",
            "cuda_available",
        }

        if not required.issubset(result):
            raise ValueError(
                f"Missing version fields: {required - result.keys()}"
            )

        return result

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Server version collection failed:\n{exc.stderr}"
        ) from exc


# ------------------------------------------------------------
# Running process discovery
# ------------------------------------------------------------

def read_process_command(pid: int) -> list[str]:
    """
    Read a process's real argument vector from /proc.

    /proc/<pid>/cmdline stores arguments separated
    by null bytes, rather than shell spaces.
    """
    data = Path(f"/proc/{pid}/cmdline").read_bytes()

    return [
        arg.decode("utf-8", errors="replace")
        for arg in data.split(b"\0")
        if arg
    ]


def find_vllm_processes() -> list[dict]:
    """
    Discover candidate vLLM serve processes.

    Only select processes whose executable environment
    matches the configured vLLM environment.
    """
    processes = []

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue

        pid = int(entry.name)

        try:
            args = read_process_command(pid)
        except (OSError, PermissionError):
            continue

        if not args:
            continue

        # Find the command sequence "vllm serve".
        is_serve = any(
            Path(args[i]).name == "vllm"
            and args[i + 1] == "serve"
            for i in range(len(args) - 1)
        )

        if not is_serve:
            continue

        # The user launches:
        # <server-python> <venv>/bin/vllm serve ...
        executable = Path(args[0])

        if executable != SERVER_PYTHON:
            continue

        processes.append({
            "pid": pid,
            "args": args,
        })

    return processes


def parse_serve_args(args: list[str]) -> dict:
    """
    Extract explicit vLLM startup arguments.

    This records configured CLI values, not every
    effective internal vLLM setting.
    """
    serve_index = next(
        i
        for i in range(len(args) - 1)
        if Path(args[i]).name == "vllm"
        and args[i + 1] == "serve"
    )

    tail = args[serve_index + 2:]

    def option(name: str) -> str | None:
        for i, arg in enumerate(tail):
            if arg == name:
                if i + 1 < len(tail):
                    return tail[i + 1]
                return None

            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]

        return None

    # Flags in vLLM can be explicitly enabled or disabled.
    if "--no-enable-prefix-caching" in tail:
        prefix_caching = False
    elif "--enable-prefix-caching" in tail:
        prefix_caching = True
    else:
        prefix_caching = None

    model = next(
        (item for item in tail if not item.startswith("-")),
        None,
    )

    max_model_len = option("--max-model-len")
    gpu_memory_utilization = option("--gpu-memory-utilization")
    port = option("--port")

    return {
        "model": model,
        "host": option("--host"),
        "port": int(port) if port is not None else None,
        "dtype": option("--dtype"),
        "prefix_caching": prefix_caching,
        "max_model_len": (
            int(max_model_len)
            if max_model_len is not None
            else None
        ),
        "gpu_memory_utilization": (
            float(gpu_memory_utilization)
            if gpu_memory_utilization is not None
            else None
        ),
        "startup_args": tail,
        "startup_command": shlex.join(args),
    }


def listening_pids(port: int) -> set[int]:
    """
    Find process IDs reported as owning a listening port.

    Uses the Linux `ss` tool. If ownership information
    cannot be read, return an empty set.
    """
    result = run_command([
        "ss",
        "-H",
        "-ltnp",
        f"sport = :{port}",
    ])

    if not result:
        return set()

    return {
        int(pid)
        for pid in re.findall(r"pid=(\d+)", result)
    }


def collect_running_server() -> dict:
    """
    Identify the running vLLM server.

    Match explicit host/port arguments first, then
    check whether the process owns the listening socket.

    Fail safely if no unique process can be identified.
    """
    candidates = []

    for process in find_vllm_processes():
        config = parse_serve_args(process["args"])

        if (
            config["host"] == SERVER_HOST
            and config["port"] == SERVER_PORT
        ):
            candidates.append({
                "pid": process["pid"],
                **config,
            })

    if len(candidates) != 1:
        return {
            "verified": False,
            "error": (
                "Expected one matching vLLM serve process, "
                f"found {len(candidates)}"
            ),
            "candidates": candidates,
        }

    server = candidates[0]

    owners = listening_pids(SERVER_PORT)

    # vLLM may use a master/server child process.
    # Therefore, socket ownership is additional evidence.
    # We do not claim verification when ownership differs
    # or is inaccessible.
    server["socket_owner_pids"] = sorted(owners)
    server["port_process_verified"] = (
        server["pid"] in owners
    )

    # This means process identity and the explicit
    # host/port command line matched.
    # It does NOT mean all settings were independently
    # confirmed through the vLLM engine.
    server["verified"] = True
    server["verification_level"] = "process_command_line"

    return server


# ------------------------------------------------------------
# Client environment
# ------------------------------------------------------------

def collect_environment() -> dict:
    """Collect all available benchmark environment metadata."""

    git_commit = run_command([
        "git",
        "-C",
        str(PROJECT_ROOT),
        "rev-parse",
        "HEAD",
    ])

    git_status = run_command([
        "git",
        "-C",
        str(PROJECT_ROOT),
        "status",
        "--porcelain",
        "--untracked-files=no",
    ])

    gpu_info = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])

    environment = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "git": {
            "commit": git_commit,
            "tracked_files_modified": (
                bool(git_status)
                if git_status is not None
                else None
            ),
        },

        "system": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "machine": platform.machine(),
        },

        "gpu": {
            "nvidia_smi": gpu_info,
        },

        "client_packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "httpx": package_version("httpx"),
            "numpy": package_version("numpy"),
            "fastapi": package_version("fastapi"),
        },
    }

    # Inspect installed server environment.
    try:
        versions = collect_server_versions()
        version_error = None

    except Exception as exc:
        versions = {}
        version_error = str(exc)

        print(
            f"[Server versions] {exc}",
            file=sys.stderr,
        )

    # Inspect running server process separately.
    try:
        running = collect_running_server()
    except Exception as exc:
        running = {
            "verified": False,
            "error": str(exc),
        }

    environment["server"] = {
        "installed_versions": versions,
        "version_collection_error": version_error,
        "running_process": running,
    }

    return environment


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    environment = collect_environment()

    print(json.dumps(
        environment,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
