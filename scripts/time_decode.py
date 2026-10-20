import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, ".")
exec(open("scripts/bench_dense.py").read().split("t = time.time()\nhasher")[0])
mx.set_cache_limit(1 << 30)
hasher = NgramHasher.load(a, root, cfg["engram_vocab_size"])
m = Transformer(a, W(), hasher)
st = m.new_state(); m.forward(st, list(range(1000, 1300)))
ts = []
for i in range(10):
    t = time.perf_counter(); mx.eval(m.forward(st, [2000 + i])); ts.append(time.perf_counter() - t)
print(f"decode compute: median {np.median(ts)*1000:.0f} ms/token  all {[round(x*1000) for x in ts]}  peak {mx.get_peak_memory()/2**30:.1f} GB")
L = int(sys.argv[1]) if len(sys.argv) > 1 else 512
toks = np.random.default_rng(0).integers(1000, 100000, size=L).tolist()
t = time.perf_counter(); mx.eval(m.forward(m.new_state(), toks, replay_from=L - 140)); dt = time.perf_counter() - t
print(f"prefill {L} bounded-replay compute: {dt:.1f}s = {L/dt:.0f} tok/s")
