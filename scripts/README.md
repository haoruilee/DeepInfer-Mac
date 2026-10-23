# Scripts

| Script | What it does |
| :--- | :--- |
| `real_check.py [N] [--noquant]` | First N converted layers of the real checkpoint vs. the official PyTorch reference, layer by layer, prefill + decode. `--noquant` disables activation rounding on both sides (logic-only check). |
| `score_api.py DIR [--max-cases K]` | Teacher-forced NLL and top-1 agreement against continuations sampled from the official API (the V4.1 vectors published in [antirez/ds4](https://github.com/antirez/ds4)), plus long-context retrieval. |
| `decode_speed.py [CACHE_GB] [--real-engram]` | End-to-end decode speed on the real model. Knobs via env: `OVERLAP`, `LOOKAHEAD`, `LOOKAHEAD_K`, `SKIP_MISS`, `PREFILL_TOPK`. |
| `profile_parts.py` | Per-component decode profile (attention projections/core, indexer, router, experts, SSD wait). |
| `trace_policies.py [record]` | Records a real routing trace and replays it against cache policies at equal capacity. |
| `bench_store.py` | Raw SSD expert-streaming throughput. |
| `real_speed.py` | Per-layer decode timing on the converted layers. |
| `make_figures.py` | Redraws the README figures (`docs/assets/*.svg`) from the measured numbers. |
| `capture_indexer.py OUT.npz [N]` | Saves the indexer's real inputs (queries, head weights, keys, candidate masks) of an N-token prompt. |
| `dosa_study.py CAPTURE.npz [SEED]` | Evaluates DoSA-style exact screening certificates on a capture (issue #8). |
