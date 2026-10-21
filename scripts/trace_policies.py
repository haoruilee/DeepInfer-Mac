"""Record the real routing trace of a decode run, then replay it against cache policies at the
same capacity (in experts per slab) to pick the best one offline."""
import collections, json, sys, time
import numpy as np
sys.path.insert(0, ".")
if sys.argv[1:2] == ["record"]:
    import mlx.core as mx
    from deepinfer_mac import LLM
    llm = LLM(expert_cache_gb=5)
    class Zeros:
        def lookup(self, layer, ids): return mx.zeros((*ids.shape, 256), dtype=mx.bfloat16)
    for b in llm.model.layers:
        if b.engram is not None: b.engram.table = Zeros()
    trace = []
    orig = type(llm.weights.experts).get_slots
    def rec(self, key, ids):
        trace.append((key, [int(i) for i in ids])); return orig(self, key, ids)
    type(llm.weights.experts).get_slots = rec
    prompts = ["Write a long paragraph about the history of tea.", "用三段话解释量子计算的基本原理。", "Write Python code for quicksort with comments."]
    for p in prompts:
        llm.reset(); ids, _ = llm.chat_prompt([{"role": "user", "content": p}])
        st, logits, _, _ = llm._prefill(ids, False); trace.append(("--decode--", []))
        tok = int(np.argmax(np.array(logits[-1])))
        for i in range(40):
            lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
    slots = {k: s.slots for k, s in llm.weights.experts.slabs.items()}
    json.dump({"trace": trace, "slots": slots}, open("/tmp/routing_trace.json", "w"))
    print("recorded", len(trace))
    sys.exit()
d = json.load(open("/tmp/routing_trace.json")); trace, slots = d["trace"], d["slots"]
groups = collections.defaultdict(list)
for k, n in slots.items(): groups[n].append(k)
def simulate(policy, scale=1.0):
    caches = {}; hits = misses = 0; decoding = False
    freq = collections.Counter(); clock = 0
    for key, ids in trace:
        if key == "--decode--": decoding = True; continue
        cap = int(slots[key] * scale); slab = tuple(sorted(k for k in slots if slots[k] == slots[key]))
        c = caches.setdefault(slab, collections.OrderedDict())
        for e in ids:
            k = (key, e); clock += 1; freq[k] += 1
            if k in c:
                hits += decoding; c.move_to_end(k); c[k] = clock
            else:
                misses += decoding
                if len(c) >= cap:
                    if policy == "lru":
                        c.popitem(last=False)
                    elif policy == "lfu":
                        victim = min(list(c)[: max(1, len(c) // 2)], key=lambda x: freq[x]); del c[victim]
                    elif policy == "lru+freq-admission":  # TinyLFU-style: only evict if newcomer is hotter
                        victim = next(iter(c))
                        if freq[k] < freq[victim] and len(c) >= cap:
                            continue
                        del c[victim]
                c[k] = clock
    return hits / max(1, hits + misses)
for pol in ("lru", "lfu", "lru+freq-admission"):
    print(f"{pol:22s} decode hit rate: " + "  ".join(f"{s}x={simulate(pol, s):.3f}" for s in (1.0, 1.5)))
