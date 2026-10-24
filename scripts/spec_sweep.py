"""Cache-size sweep with idle-window speculation off/on interleaved in one process per cache size;
raw data to OUT.json. Usage: spec_sweep.py OUT.json CACHE1,CACHE2,... [rounds] [tokens_per_round]"""
import json, subprocess, sys
from pathlib import Path
out, sizes = sys.argv[1], sys.argv[2].split(",")
rounds = sys.argv[3] if len(sys.argv) > 3 else "4"
per = sys.argv[4] if len(sys.argv) > 4 else "6"
res = {}
for c in sizes:
    tmp = f"{out}.{c}.json"
    subprocess.run([sys.executable, str(Path(__file__).with_name("spec_ablation.py")), tmp, c, rounds, per, "overlap,idle-window"], check=True)
    res[c] = json.loads(Path(tmp).read_text())
    Path(out).write_text(json.dumps(res))
