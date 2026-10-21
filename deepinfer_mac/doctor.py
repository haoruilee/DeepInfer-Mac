"""`python -m deepinfer_mac doctor`: is this Mac ready, and how is the conversion going?"""

import json
import shutil
import subprocess
from pathlib import Path


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def conversion_status(root: Path) -> tuple[int, int, float]:
    """(tasks done, tasks total, bytes still to download)."""
    from .convert import remote as R

    c = R.Converter.__new__(R.Converter)
    c.idx = json.loads(R.INDEX.read_text())
    c.inf = json.loads((root / "inference_config.json").read_text())
    done = set((root / "progress.log").read_text().split()) if (root / "progress.log").exists() else set()
    total = left = 0
    tasks = c.tasks()
    for tid, kind, p in tasks:
        n = (p["r1"] - p["r0"]) * 264 if kind == "EG" else sum(m["end"] - m["start"] for m in p.values())
        total += 1
        if tid not in done:
            left += n
    return len(done & {t[0] for t in tasks}), total, left


def main(root: Path):
    from .weights import available_memory, default_expert_budget, total_memory

    print(f"model dir: {root}")
    ok = True
    if not (root / "inference_config.json").exists():
        print("  ✗ not converted yet: run `python -m deepinfer_mac convert`")
        return
    done, total, left = conversion_status(root)
    state = "complete" if done == total else f"{done}/{total} tasks, {_gb(left)} left to download"
    print(f"  conversion: {state}")
    ok &= done == total
    free = shutil.disk_usage(root).free
    print(f"  disk free:  {_gb(free)}" + ("  ✗ not enough for the rest" if free < left * 0.7 + 8e9 else ""))
    print(f"  memory:     {_gb(total_memory())} total, {_gb(available_memory())} available now")
    if done == total or (root / "dense" / "globals.safetensors").exists():
        budget = default_expert_budget(root)
        hint = "" if budget >= 5 << 30 else "  (close memory-heavy apps for a bigger cache = faster decode)"
        print(f"  expert cache would be: {budget / 2**30:.1f} GB{hint}")
    top = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True).stdout.splitlines()
    top = sorted((int(l.split(None, 1)[0]), l.split(None, 1)[1].rsplit("/", 1)[-1]) for l in top if l.strip())[-5:][::-1]
    print("  largest processes: " + ", ".join(f"{n} {r / 2**20:.1f} GB" for r, n in top))
    proxy = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True).stdout
    if "utun" in proxy:
        print("  note: default route is a VPN/TUN interface; the converter reaches ModelScope directly anyway")
    print("ready to run" if ok else "not ready yet")
