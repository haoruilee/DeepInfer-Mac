"""Plots for the technical report, from measured values (repository logs and paper/figures/data).
Run: uv run --with matplotlib python paper/figures/make_plots.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
# pastel fills with darker edges and hatching, as in the DoSA figures
YEL, YEL_E = "#f7e08f", "#b8962b"
BLU, BLU_E = "#a9c4eb", "#4a72b0"
GRN, GRN_E = "#b8dca0", "#5a8f3c"
GRY, GRY_E = "#d9d9d9", "#7f7f7f"
ORG, ORG_E = "#f6c19c", "#c0652a"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "hatch.linewidth": 0.5, "pdf.fonttype": 42,
                     "axes.titlesize": 8.5, "legend.fontsize": 7, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.linewidth": 0.4, "grid.alpha": 0.5, "axes.axisbelow": True})


def bar(ax, x, h, fill, edge, hatch, **k):
    return ax.bar(x, h, color=fill, edgecolor=edge, hatch=hatch, linewidth=0.7, **k)


def labels(ax, bars, fmt="{:.2f}", dy=0.0):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=6.5)


def budget():
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 1.75))
    for ax, parts, unit, title in (
        (axes[0], [("SSD wait", 0.55, ORG, ORG_E, "//"), ("compute", 0.25, BLU, BLU_E, "\\\\")], "s", "One decode step (0.80 s)"),
        (axes[1], [("resident weights", 8.4, BLU, BLU_E, "\\\\"), ("expert cache", 6.6, YEL, YEL_E, "//"), ("macOS + apps", 9.0, GRY, GRY_E, "")], "GB", "Unified memory (24 GB)"),
    ):
        x = 0
        for name, v, c, e, h in parts:
            ax.barh(0, v, left=x, color=c, edgecolor=e, hatch=h, height=0.62, linewidth=0.7)
            ax.text(x + v / 2, 0, f"{name}: {v:g} {unit}", ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            x += v
        ax.set_xlim(0, x)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, loc="left", pad=2)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(OUT / "budget.pdf")


def api_cases():
    prompt = [127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 16383, 16384, 16385]
    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    b1 = bar(ax, range(5), [78.1, 84.4, 100, 90.6, 90.6], BLU, BLU_E, "\\\\\\", width=0.7, label="open-ended continuation")
    b2 = bar(ax, range(5, 17), [100.0] * 12, GRN, GRN_E, "///", width=0.7, label="needle retrieval")
    labels(ax, b1, "{:.1f}", 0.5)
    ax.set_xticks(range(17), ["prose", "code", "arith.", "prose", "Italian"] + [str(p) for p in prompt], rotation=55,
                  fontsize=6.5, ha="right", rotation_mode="anchor")
    ax.set_ylim(60, 112)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_ylabel("Top-1 agreement (%)")
    ax.set_xlabel("case (retrieval: prompt length in tokens)", fontsize=7)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / "api_cases.pdf")


def _timeline(ax, ev, layers, title):
    ev = [e for e in ev if e["layer"] in layers]
    t0 = min(e["t0"] for e in ev if e["kind"] == "route") - 0.008
    threads = sorted({e["thread"] for e in ev if e["kind"] == "read"}, key=lambda s: (("spec" in s), int(s.split("_")[-1])))
    rows = ["GPU"] + [("spec" if "spec" in t else "I/O") + f" {t.split('_')[-1]}" for t in threads]
    ms = lambda t: (t - t0) * 1e3  # noqa: E731
    prev_end = None
    for L in layers:
        es = [e for e in ev if e["layer"] == L]
        route = [e for e in es if e["kind"] == "route"][0]
        start = ms(prev_end) if prev_end is not None else 0.0
        ax.barh(0, ms(route["t0"]) - start, left=start, color=GRY, edgecolor=GRY_E, height=0.7, linewidth=0.6)
        waits = [e for e in es if e["kind"] == "wait"]
        for g in [e for e in es if e["kind"] == "gpu"]:
            a, b = ms(g["t0"]), ms(g["t1"])
            for w in waits:
                if w["t0"] >= g["t0"] - 1e-4 and w["t1"] <= g["t1"] + 1e-4 and w["t1"] > w["t0"]:
                    a = max(a, ms(w["t1"]))
            ax.barh(0, max(b - a, 0.3), left=a, color=BLU, edgecolor=BLU_E, hatch="\\\\\\", height=0.7, linewidth=0.6)
        end = max(e.get("t1", e["t0"]) for e in es)
        ax.text((ms(route["t0"]) + ms(end)) / 2, len(rows) + 0.25, f"layer {int(L[1:])}", ha="center", va="center", fontsize=6.3)
        ax.axvline(ms(route["t0"]), color="#666", lw=0.5, ls=":")
        prev_end = end
    for r in [e for e in ev if e["kind"] == "read"]:
        y = 1 + threads.index(r["thread"])
        spec = r.get("spec", False)
        ax.barh(y, ms(r["t1"]) - ms(r["t0"]), left=ms(r["t0"]), color=GRN if spec else ORG, edgecolor=GRN_E if spec else ORG_E,
                hatch="xx" if spec else "///", height=0.7, linewidth=0.6)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.set_ylim(len(rows) + 0.7, -0.6)
    ax.grid(False)
    ax.set_title(title, loc="left")
    return ms(prev_end)


def timeline():
    from matplotlib.patches import Patch
    d = json.loads((OUT / "data" / "trace_layers.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.6), sharex=True)
    x1 = _timeline(axes[0], d["off"]["events"], d["off"]["layers"], "(a) Without speculation")
    x2 = _timeline(axes[1], d["spec"]["events"], d["spec"]["layers"], "(b) With idle-window speculation ($k=3$)")
    axes[1].set_xlim(0, max(x1, x2) + 2)
    axes[1].set_xlabel("time (ms)")
    fig.legend(handles=[Patch(fc=GRY, ec=GRY_E, label="attention, mixers, router"), Patch(fc=BLU, ec=BLU_E, hatch="\\\\\\", label="expert compute"),
                        Patch(fc=ORG, ec=ORG_E, hatch="///", label="demand read"), Patch(fc=GRN, ec=GRN_E, hatch="xx", label="speculative read")],
               frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.4)
    fig.savefig(OUT / "timeline.pdf")


def sweep():
    """Decode speed vs. expert cache size, speculation off/on interleaved, and the cost model."""
    d = json.loads((OUT / "data" / "sweep2.json").read_text())
    rows = d["rows"]
    c = [r["cache"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95))
    ax = axes[0]
    ax.errorbar(c, [r["tok_ov"] for r in rows], yerr=[r["tok_ci_ov"] for r in rows], marker="o", color=BLU_E, lw=1.2, ms=3.5, capsize=2, label="no speculation")
    ax.errorbar(c, [r["tok_iw"] for r in rows], yerr=[r["tok_ci_iw"] for r in rows], marker="s", color=GRN_E, lw=1.2, ms=3.5, capsize=2, label="idle-window spec. ($k=3$)")
    ax.plot(c, [1 / r["pred_ov"] for r in rows], ls="--", color="black", lw=0.8, label="model fit")
    ax.plot(c, [1 / r["pred_iw"] for r in rows], ls=":", color="black", lw=0.9, label="model, spec. bytes $\\times$" + f"{d['alpha']:.2f}")
    ax.set_xlabel("expert cache (GB)")
    ax.set_ylabel("decode (tokens/s)")
    ax.set_title("(a) Decode speed")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")
    ax = axes[1]
    ax.plot(c, [r["wait_ov"] for r in rows], marker="o", color=BLU_E, lw=1.2, ms=3.5)
    ax.plot(c, [r["wait_iw"] for r in rows], marker="s", color=GRN_E, lw=1.2, ms=3.5)
    ax.set_xlabel("expert cache (GB)")
    ax.set_ylabel("SSD wait (ms/token)")
    ax.set_title("(b) Time waiting for the SSD")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "sweep.pdf")


def parallel(prefill):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [0.95, 1.6, 0.7]})
    ax = axes[0]
    th = [1, 2, 4, 8, 16, 32]
    bw = [2.70, 2.77, 3.01, 3.00, 2.98, 2.92]  # clean machine
    bs = bar(ax, range(len(th)), bw, YEL, YEL_E, "///", width=0.65)
    for i, b in enumerate(bs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(len(th)), th)
    ax.set_xlabel("reads in flight")
    ax.set_ylabel("GB/s")
    ax.set_ylim(0, 3.6)
    ax.set_title("(a) SSD concurrency")
    ax = axes[1]
    ab = json.loads((OUT / "data" / "ablation_ratios.json").read_text())
    names = ["overlap", "naive-prefetch", "spec-neither", "spec-cold-only", "spec-wait-only", "idle-window"]
    shown = ["layer\noverlap", "naive\nprefetch", "guesses,\nno wait", "cold\nonly", "wait\nonly", "idle-\nwindow"]
    rel = [ab[n][0] for n in names]
    good = [r > 1.02 for r in rel]
    bs = ax.bar(range(len(rel)), rel, width=0.65, linewidth=0.7, color=[GRN if g else BLU for g in good],
                edgecolor=[GRN_E if g else BLU_E for g in good], hatch="\\\\\\", yerr=[ab[n][1] for n in names],
                error_kw=dict(lw=0.7, capsize=1.5))
    for i, b in enumerate(bs):
        ax.text(b.get_x() + b.get_width() / 2, rel[i] + ab[names[i]][1] + 0.02, f"{rel[i]:.2f}", ha="center", fontsize=6)
    ax.axhline(1.0, color="black", lw=0.6, ls="--")
    ax.set_xticks(range(len(rel)), shown, fontsize=5.6)
    ax.set_ylim(0, 1.38)
    ax.set_ylabel("speed-up over serial")
    ax.set_title("(b) Decode schedules (5 GB)")
    ax = axes[2]
    on, off = prefill
    med = lambda xs: sorted(xs)[len(xs) // 2 - 1] / 2 + sorted(xs)[len(xs) // 2] / 2  # noqa: E731
    b = bar(ax, [0, 1], [sum(off) / len(off), sum(on) / len(on)], GRN, GRN_E, "xx", width=0.6)
    for i, xs in enumerate((off, on)):
        ax.scatter([i] * len(xs), xs, s=7, color="black", zorder=3)
    for bb in b:
        ax.text(bb.get_x() + bb.get_width() / 2, 4, f"{bb.get_height():.0f}", ha="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.set_xticks([0, 1], ["serial", "pipelined"])
    ax.set_ylabel("prefill time (s)")
    ax.set_ylim(0, 62)
    ax.set_title("(c) Prefill, 512 tok.")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "parallel.pdf")


def indexer():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    names = ["L2\n255×16K", "L14\n255×16K", "L24\n128×32K", "L36\n128×32K"]
    old, new = [128.0, 126.6, 125.3, 128.1], [70.5, 60.9, 35.8, 66.6]  # paper/logs/bench_kernels.log
    x = range(len(names))
    b1 = bar(ax, [i - 0.19 for i in x], old, GRY, GRY_E, "", width=0.38, label="FP32 einsum + 3 passes")
    b2 = bar(ax, [i + 0.19 for i in x], new, BLU, BLU_E, "\\\\\\", width=0.38, label="BF16 GEMM + fused kernel")
    labels(ax, b1, "{:.0f}", 2)
    labels(ax, b2, "{:.0f}", 2)
    ax.set_xticks(list(x), names, fontsize=6.3)
    ax.set_ylabel("ms per query block")
    ax.set_ylim(0, 275)
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.3)
    ax.set_title("(a) Exact reformulation")
    ax = axes[1]
    layers = [2, 8, 14, 24, 28, 32, 36]
    cost = [85, 87, 91, 85, 88, 83, 81]  # scripts/dosa_study.py, seed 4096, 17 queries per layer
    x = range(len(layers))
    bs = bar(ax, x, cost, YEL, YEL_E, "///", width=0.62)
    for i, v in enumerate(cost):
        ax.text(i, v + 1.5, f"{v}", ha="center", fontsize=6.5)
    ax.axhline(100, color="black", lw=0.6, ls="--")
    ax.text(-0.45, 102, "full scoring", ha="left", va="bottom", fontsize=6.3)
    ax.set_xticks(list(x), [f"L{l}" for l in layers], fontsize=6.5)
    ax.set_ylabel("cost vs. full scoring (%)")
    ax.set_ylim(0, 140)
    ax.set_title("(b) Exact screening (DoSA-style)")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "indexer.pdf")


if __name__ == "__main__":
    pf = json.loads((OUT / "data" / "prefill_overlap.json").read_text())
    budget(), api_cases(), timeline(), parallel((pf["on"], pf["off"])), indexer(), sweep()
