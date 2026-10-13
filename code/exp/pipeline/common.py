"""Shared plumbing: git-style hashes, seeds, GPU guard, run metadata, result writing.

Every stage of the P1 pipeline writes its outputs together with a `<stage>.meta.json` produced by
`write_meta`. The meta file records the full config, a git-style config hash, a hash of the
pipeline source, the seeds, input/output file hashes and the software/hardware environment, so any
number in the paper can be traced back to the exact code and settings that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PIPELINE_DIR = Path(__file__).resolve().parent
EXP_DIR = PIPELINE_DIR.parent

# hostA policy (TEAM_RULES §4): GPUs 6-7 belong to P1. 0-3 are another team's; 4 -> P2, 5 -> P3.
P1_ALLOWED_GPUS = {6, 7}
# hostA: P1 lanes are 0-3 (coordinator, 2026-10-04); 6/7 belong to P2. Default is set by hostname in guard_gpus(); do not widen via the environment.


def trust_remote() -> bool:
    """P1_TRUST_REMOTE=1 allows trust_remote_code for official repos that ship config/model code (XiaomiMiMo/MiMo-*); off by default."""
    return os.environ.get("P1_TRUST_REMOTE") == "1"


# ----------------------------------------------------------------------------- hashing / seeds
def git_blob_hash(data: bytes) -> str:
    """Same object id as `git hash-object` for a blob with this content."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_hash(path: os.PathLike) -> str:
    return git_blob_hash(Path(path).read_bytes())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(cfg: Dict[str, Any]) -> str:
    """git-style hash of the canonical JSON of a config (volatile keys must not be inside cfg)."""
    return git_blob_hash(canonical_json(cfg).encode("utf-8"))


def code_hash(pipeline_dir: Optional[os.PathLike] = None) -> str:
    """git-style hash over all pipeline *.py files (tests excluded), order-independent."""
    root = Path(pipeline_dir) if pipeline_dir else PIPELINE_DIR
    lines = []
    for p in sorted(root.glob("*.py")):
        lines.append("%s %s" % (p.name, file_hash(p)))
    return git_blob_hash("\n".join(lines).encode())


def derive_seed(base: int, *keys: Any) -> int:
    """Deterministic 31-bit seed from a base seed and arbitrary keys (used per request)."""
    h = hashlib.sha256(("%d|" % base + "|".join(str(k) for k in keys)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


# ----------------------------------------------------------------------------- GPU guard
def guard_gpus(allowed: Iterable[int] = P1_ALLOWED_GPUS) -> List[int]:
    """Refuse to run on GPUs outside P1's allocation. CUDA_VISIBLE_DEVICES must be set explicitly.

    Only enforced on hostA (hostname contains 'dxm'); on other hosts (Slurm, laptop) it returns [].
    """
    host = socket.gethostname().lower()
    if "dxm" not in host:
        return []
    if os.environ.get("P1_ALLOWED_GPUS"):
        allowed = {int(x) for x in os.environ["P1_ALLOWED_GPUS"].split(",") if x.strip()}
    elif "dxm-25" in host or "hostA" in host:
        allowed = {0, 1, 2, 3}   # coordinator lane assignment for P1 on hostA (6/7 belong to P2); the module default {6, 7} is the old hostA lane
    # P1_ALLOWED_GPUS must never be widened beyond the coordinator's lanes by a caller; ask the coordinator for extra lanes
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        raise SystemExit("CUDA_VISIBLE_DEVICES is unset on %s. Set it to 6, 7 or 6,7 (P1 GPUs only)." % host)
    ids = [int(x) for x in cvd.split(",") if x.strip() != ""]
    bad = [i for i in ids if i not in set(allowed)]
    if bad:
        raise SystemExit("GPU(s) %s are not allocated to P1 (allowed: %s)." % (bad, sorted(allowed)))
    return ids


# ----------------------------------------------------------------------------- environment
def _pkg_versions(names: Iterable[str]) -> Dict[str, str]:
    import importlib.metadata as md

    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = "not-installed"
    return out


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "packages": _pkg_versions(["vllm", "torch", "transformers", "numpy", "scipy", "math-verify",
                                   "matplotlib", "pandas", "scikit-learn"]),
    }
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        if q.returncode == 0:
            cvd = info["cuda_visible_devices"]
            rows = [r.strip() for r in q.stdout.strip().splitlines()]
            if cvd:
                keep = {c.strip() for c in cvd.split(",")}
                rows = [r for r in rows if r.split(",")[0].strip() in keep]
            info["gpus"] = rows
    except Exception:
        pass
    try:
        rev = subprocess.run(["git", "-C", str(EXP_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        info["git_head"] = rev.stdout.strip() if rev.returncode == 0 else "no-git-repo"
    except Exception:
        info["git_head"] = "no-git-repo"
    return info


# ----------------------------------------------------------------------------- io
def read_jsonl(path: os.PathLike) -> List[Dict[str, Any]]:
    import gzip

    p = str(path)
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: os.PathLike, rows: Iterable[Dict[str, Any]]) -> int:
    import gzip

    p = str(path)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.endswith(".gz") else open
    n = 0
    with opener(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        f.write("\n")


def read_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_meta(path: os.PathLike, stage: str, config: Dict[str, Any], seeds: Dict[str, Any],
               inputs: Iterable[os.PathLike] = (), outputs: Iterable[os.PathLike] = (),
               extra: Optional[Dict[str, Any]] = None, t_start: Optional[float] = None) -> Dict[str, Any]:
    """Write `<stage>.meta.json`: config + hashes + seeds + env. Returns the dict."""
    meta = {
        "stage": stage,
        "config": config,
        "config_hash": config_hash(config),
        "code_hash": code_hash(),
        "seeds": seeds,
        "inputs": {str(p): file_hash(p) for p in inputs if Path(p).exists()},
        "outputs": {str(p): file_hash(p) for p in outputs if Path(p).exists()},
        "env": env_info(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if t_start is not None:
        meta["wall_seconds"] = round(time.time() - t_start, 1)
    if extra:
        meta["extra"] = extra
    write_json(path, meta)
    return meta
