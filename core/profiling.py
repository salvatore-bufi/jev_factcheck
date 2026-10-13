"""Measurement helpers: environment description, a background NVML sampler, GPU-busy check, RSS.

The sampler writes one record every `interval` seconds to telemetry.jsonl: power, utilisation,
memory, temperature, clocks and throttle reasons of the benchmark GPU, plus the resident memory
of this process tree (vLLM runs its engine in a child process). Each record carries the current
phase (load, warmup, latency, throughput, main), so energy and peaks can be split per phase.
"""
import json
import os
import platform
import socket
import subprocess
import threading
import time
from importlib import metadata
from pathlib import Path

from . import ROOT

PACKAGES = ("torch", "transformers", "vllm", "llm2jev", "contrastive-lm", "numpy", "pyarrow", "scikit-learn",
            "nvidia-ml-py", "huggingface-hub", "safetensors", "gguf", "requests")


def physical_gpu():
    """The physical GPU index this process runs on (first entry of CUDA_VISIBLE_DEVICES), or None."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return 0
    first = visible.split(",")[0].strip()
    return int(first) if first.isdigit() else None


def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def environment():
    """Hardware and software description for run.json."""
    info = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or None, "cpu_count": os.cpu_count(),
            "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                    "dirty": bool(_git("status", "--porcelain"))},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": {}}
    try:
        with open("/proc/meminfo") as handle:
            info["ram_total_gb"] = round(int(handle.readline().split()[1]) / 1e6, 1)
        with open("/proc/cpuinfo") as handle:
            info["cpu"] = next((l.split(":", 1)[1].strip() for l in handle if l.startswith("model name")), info["cpu"])
    except OSError:
        pass
    for name in PACKAGES:
        try:
            info["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    nv = _nvml()
    index = physical_gpu()
    if nv is not None and index is not None:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        name = nv.nvmlDeviceGetName(handle)
        info["gpu"] = {"index": index, "name": name.decode() if isinstance(name, bytes) else name,
                       "memory_total_mb": nv.nvmlDeviceGetMemoryInfo(handle).total // 2**20,
                       "driver": _decode(nv.nvmlSystemGetDriverVersion()),
                       "cuda_driver": nv.nvmlSystemGetCudaDriverVersion(),
                       "power_limit_w": _safe(lambda: nv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000),
                       "pcie_gen_max": _safe(lambda: nv.nvmlDeviceGetMaxPcieLinkGeneration(handle)),
                       "pcie_width_max": _safe(lambda: nv.nvmlDeviceGetMaxPcieLinkWidth(handle))}
    return info


def _decode(x):
    return x.decode() if isinstance(x, bytes) else x


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def other_gpu_processes():
    """Other processes on the benchmark GPU: {"compute": [...], "graphics": [...]}.

    Compute processes (another model, a training job) distort latency and memory, so the runner refuses
    to start when there is one. Graphics processes (a desktop compositor on a workstation) are only recorded.
    """
    nv = _nvml()
    index = physical_gpu()
    out = {"compute": [], "graphics": []}
    if nv is None or index is None:
        return out
    handle = nv.nvmlDeviceGetHandleByIndex(index)
    mine = process_tree(os.getpid())
    for kind, getter in (("compute", "nvmlDeviceGetComputeRunningProcesses"), ("graphics", "nvmlDeviceGetGraphicsRunningProcesses")):
        try:
            for p in getattr(nv, getter)(handle):
                if p.pid not in mine:
                    out[kind].append({"pid": p.pid, "used_mb": (p.usedGpuMemory or 0) // 2**20, "name": _proc_name(p.pid)})
        except Exception:
            pass
    return out


def _proc_name(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return None


def process_tree(pid):
    """pid and all its descendants (from /proc)."""
    children = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(entry.name))
            except (OSError, IndexError, ValueError):
                continue
    tree, stack = set(), [pid]
    while stack:
        p = stack.pop()
        tree.add(p)
        stack += children.get(p, [])
    return tree


def rss_mb(pids):
    total = 0
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except OSError:
            continue
    return total / 1024


class Telemetry:
    """Background sampler; `phase(name)` tags the following samples; `summary()` aggregates per phase."""

    def __init__(self, path, interval=0.2):
        self.path, self.interval = Path(path), interval
        self.nv = _nvml()
        index = physical_gpu()
        self.handle = self.nv.nvmlDeviceGetHandleByIndex(index) if self.nv is not None and index is not None else None
        self.current = "start"
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def phase(self, name):
        self.current = name

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _sample(self):
        rec = {"t": time.time(), "phase": self.current, "rss_mb": round(rss_mb(process_tree(os.getpid())), 1)}
        if self.handle is not None:
            nv, h = self.nv, self.handle
            util = _safe(lambda: nv.nvmlDeviceGetUtilizationRates(h))
            rec.update(power_w=_safe(lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000),
                       gpu_util=getattr(util, "gpu", None), mem_util=getattr(util, "memory", None),
                       mem_used_mb=_safe(lambda: nv.nvmlDeviceGetMemoryInfo(h).used // 2**20),
                       temp_c=_safe(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                       sm_clock_mhz=_safe(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                       mem_clock_mhz=_safe(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                       pcie_gen=_safe(lambda: nv.nvmlDeviceGetCurrPcieLinkGeneration(h)),
                       throttle=_safe(lambda: nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)))
        return rec

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            while not self._stop.is_set():
                rec = self._sample()
                self.samples.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                self._stop.wait(self.interval)


def summarize_telemetry(samples):
    """Per phase: duration, energy (J, trapezoid over power), mean/max power, mean util, peak memory, max temp."""
    out = {}
    by_phase = {}
    for s in samples:
        by_phase.setdefault(s["phase"], []).append(s)
    for phase, rows in by_phase.items():
        energy = 0.0
        for a, b in zip(rows, rows[1:]):
            if a.get("power_w") is not None and b.get("power_w") is not None and b["t"] - a["t"] < 5:
                energy += (a["power_w"] + b["power_w"]) / 2 * (b["t"] - a["t"])
        pw = [r["power_w"] for r in rows if r.get("power_w") is not None]
        ut = [r["gpu_util"] for r in rows if r.get("gpu_util") is not None]
        mem = [r["mem_used_mb"] for r in rows if r.get("mem_used_mb") is not None]
        tmp = [r["temp_c"] for r in rows if r.get("temp_c") is not None]
        clk = [r["sm_clock_mhz"] for r in rows if r.get("sm_clock_mhz") is not None]
        out[phase] = {"samples": len(rows), "seconds": round(rows[-1]["t"] - rows[0]["t"], 3) if len(rows) > 1 else 0,
                      "energy_j": round(energy, 1),
                      "power_mean_w": round(sum(pw) / len(pw), 1) if pw else None, "power_max_w": max(pw) if pw else None,
                      "gpu_util_mean": round(sum(ut) / len(ut), 1) if ut else None,
                      "gpu_mem_peak_mb": max(mem) if mem else None,
                      "rss_peak_mb": max(r["rss_mb"] for r in rows),
                      "temp_max_c": max(tmp) if tmp else None,
                      "sm_clock_mean_mhz": round(sum(clk) / len(clk)) if clk else None,
                      "sm_clock_min_mhz": min(clk) if clk else None}
    return out
