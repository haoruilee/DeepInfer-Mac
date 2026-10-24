"""Draw the README figures (docs/assets/*.svg) from the measured numbers (paper/figures/data, paper/logs).

The SVGs follow the viewer's light/dark setting (prefers-color-scheme), use system fonts only, and
are plain text so they diff and stay small. GitHub shows them at about 65% of their 1280-px width,
so no text is smaller than 16 px. Run: python scripts/make_figures.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
DATA = ROOT / "paper" / "figures" / "data"

# on-disk sizes (GiB, du of the converted checkpoint) and the 5.5 GB-cache operating point of the
# interleaved sweep (paper/logs/sweep2.json*): speculation on, cache that keeps 9 GB for macOS
TOTAL_GB, RESIDENT_GB, EXPERTS_GB, ENGRAM_GB = 338, 7.6, 208, 114
SWEEP = {r["cache"]: r for r in json.loads((DATA / "sweep2.json").read_text())["rows"]}
OP = SWEEP[5.5]
DECODE_TOKS = OP["tok_iw"]                      # 1.23 tok/s
TOKEN_S, SSD_WAIT_S = OP["t_iw"], OP["wait_iw"] / 1e3
DEMAND_GB, SPEC_GB = OP["demand_iw"], OP["spec_iw"]
OPEN_ENDED_AGREEMENT = 86.0                     # top-1 vs. the API over 129 open-ended tokens

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'PingFang SC', 'Noto Sans Mono CJK SC', monospace"

THEME = """
  .card { fill: #fcfcfb; stroke: #e4e3df; }
  .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #7a7974; }
  .ram { fill: #2a78d6; } .read { fill: #eb6834; } .spec { fill: #1baf7a; } .disk { fill: #dcdad3; } .attn { fill: #b9b7af; }
  .gpu { fill: #e9f1fc; stroke: #2a78d6; } .ssdbox { fill: #fdf0ea; stroke: #eb6834; }
  .box { fill: #ffffff; stroke: #c9c7bf; } .wire { stroke: #8a8984; } .ah { fill: #8a8984; }
  .wire-read { stroke: #eb6834; } .ah-read { fill: #eb6834; } .wire-spec { stroke: #1baf7a; } .ah-spec { fill: #1baf7a; }
  .accent { fill: #1c5cab; } .accent2 { fill: #c4501f; } .accent3 { fill: #11805a; }
  .grid { stroke: #e4e3df; } .axis { stroke: #8a8984; }
  .l-ov { stroke: #2a78d6; } .l-iw { stroke: #1baf7a; } .d-ov { fill: #2a78d6; } .d-iw { fill: #1baf7a; }
  .ring { stroke: #fcfcfb; }
  @media (prefers-color-scheme: dark) {
    .card { fill: #161b22; stroke: #30363d; }
    .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .t3 { fill: #94938c; }
    .ram { fill: #3987e5; } .read { fill: #d95926; } .spec { fill: #199e70; } .disk { fill: #3a3a37; } .attn { fill: #5b5a55; }
    .gpu { fill: #14243a; stroke: #3987e5; } .ssdbox { fill: #2e1c14; stroke: #d95926; }
    .box { fill: #0d1117; stroke: #484f58; } .wire { stroke: #8b949e; } .ah { fill: #8b949e; }
    .wire-read { stroke: #e0703f; } .ah-read { fill: #e0703f; } .wire-spec { stroke: #2bbd88; } .ah-spec { fill: #2bbd88; }
    .accent { fill: #86b6ef; } .accent2 { fill: #f08a5d; } .accent3 { fill: #5fd3a6; }
    .grid { stroke: #30363d; } .axis { stroke: #8b949e; }
    .l-ov { stroke: #3987e5; } .l-iw { stroke: #2bbd88; } .d-ov { fill: #3987e5; } .d-iw { fill: #2bbd88; }
    .ring { stroke: #161b22; }
  }
"""


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{title}">\n<title>{title}</title>\n<style>{THEME}  text {{ font-family: {SANS}; }}\n'
            f'  .mono {{ font-family: {MONO}; }}\n</style>\n{body}\n</svg>\n')


def card(w, h):
    return f'<rect class="card" x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="16"/>'


def legend(x, y, items, gap=34):
    out = []
    for cls, label in items:
        out.append(f'<rect class="{cls}" x="{x}" y="{y - 13}" width="16" height="16" rx="4"/>'
                   f'<text class="t2" x="{x + 24}" y="{y}" font-size="18">{label}</text>')
        x += 24 + 9.6 * len(label) + gap
    return "".join(out)


# ------------------------------------------------------------------------------ hero

def hero():
    """One square per GB of converted weights: resident, read per token (needed / read ahead), on SSD."""
    W, H = 1280, 520
    cols, cell, gap = 26, 17, 4
    n_ram, n_dem, n_spec = round(RESIDENT_GB), max(1, round(DEMAND_GB)), max(1, round(SPEC_GB))
    gx, gy = 680, 100
    sq = []
    for i in range(TOTAL_GB):
        r, c = divmod(i, cols)
        cls = "ram" if i < n_ram else "read" if i < n_ram + n_dem else "spec" if i < n_ram + n_dem + n_spec else "disk"
        sq.append(f'<rect class="{cls}" x="{gx + c * (cell + gap)}" y="{gy + r * (cell + gap)}" width="{cell}" height="{cell}" rx="3"/>')
    rows = -(-TOTAL_GB // cols)
    ly = gy + rows * (cell + gap) + 30
    leg = (f'<rect class="ram" x="{gx}" y="{ly - 13}" width="16" height="16" rx="4"/><text class="t2" x="{gx + 24}" y="{ly}" font-size="18">{RESIDENT_GB:.1f} GB in memory</text>'
           f'<rect class="read" x="{gx + 240}" y="{ly - 13}" width="16" height="16" rx="4"/><text class="t2" x="{gx + 264}" y="{ly}" font-size="18">read per token</text>'
           f'<rect class="spec" x="{gx}" y="{ly + 17}" width="16" height="16" rx="4"/><text class="t2" x="{gx + 24}" y="{ly + 30}" font-size="18">read ahead in idle SSD time</text>'
           f'<rect class="disk" x="{gx + 300}" y="{ly + 17}" width="16" height="16" rx="4"/><text class="t2" x="{gx + 324}" y="{ly + 30}" font-size="18">on the SSD</text>')
    stats = [(f"{DECODE_TOKS:.2f}", "tok/s decode"), (f"{OPEN_ENDED_AGREEMENT:.0f}%", "top-1 match with the API"), ("16K", "token retrieval")]
    st = "".join(f'<text class="accent" x="{64 + j * 200}" y="440" font-size="44" font-weight="700">{v}</text>'
                 f'<text class="t2" x="{64 + j * 200}" y="470" font-size="17">{lab}</text>' for j, (v, lab) in enumerate(stats))
    body = f"""{card(W, H)}
<text class="t3" x="64" y="104" font-size="19" font-weight="600" letter-spacing="1.5">DEEPSEEK-V4.1-FLASH · MLX · APPLE SILICON</text>
<text class="t1" x="64" y="172" font-size="60" font-weight="700" letter-spacing="-1">DeepInfer-Mac</text>
<text class="t1" x="64" y="232" font-size="33" font-weight="600">A 754B-parameter model</text>
<text class="t1" x="64" y="274" font-size="33" font-weight="600">on a 24 GB MacBook Air.</text>
<text class="t2" x="64" y="324" font-size="20">{TOTAL_GB} GB of weights, {RESIDENT_GB:.1f} GB in memory.</text>
<text class="t2" x="64" y="354" font-size="20">Experts stream from the SSD; the next layer's</text>
<text class="t2" x="64" y="384" font-size="20">likely experts are read while the GPU computes.</text>
{st}
<text class="t3" x="{gx}" y="80" font-size="17">each square = 1 GB of converted weights</text>
{''.join(sq)}
{leg}"""
    return svg(W, H, body, f"DeepInfer-Mac: {TOTAL_GB} GB of DeepSeek-V4.1-Flash weights, {RESIDENT_GB:.1f} GB in memory")


# ------------------------------------------------------------------------------ chat transcript

TRANSCRIPT = [
    ("you", "用一句话介绍一下杭州。"),
    ("ds", "杭州是中国浙江省的省会，以西湖美景、深厚的历史文化底蕴和蓬勃发展的数字经济而闻名。"),
    ("meta", "[decode 1.23 tok/s]"),
    ("you", "What is 17 * 23? Answer with the number only."),
    ("ds", "391"),
    ("you", "Write a haiku about the sea."),
    ("ds", "Waves crash on the shore,"),
    ("cont", "Salt and endless blue expanse,"),
    ("cont", "Moon pulls the tide home."),
    ("you", "/img carrots.jpeg What is in this image? One sentence."),
    ("ds", "A small pile of four fresh orange carrots with green stems rests on a white background."),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def terminal():
    """The README transcript (real output, greedy decoding) as a terminal window. Always dark."""
    W, lh, top = 1280, 36, 104
    H = top + lh * (len(TRANSCRIPT) + 1) + 30
    lines, y = [], top
    lines.append(f'<text class="mono" x="40" y="{y}" font-size="20" fill="#8b949e">$ <tspan fill="#e6edf3">python -m deepinfer_mac chat</tspan></text>')
    y += lh + 8
    for who, text in TRANSCRIPT:
        if who == "you":
            t = f'<tspan fill="#79c0ff" font-weight="700">you&gt; </tspan><tspan fill="#e6edf3">{esc(text)}</tspan>'
        elif who == "ds":
            t = f'<tspan fill="#ffa657" font-weight="700">ds&gt;  </tspan><tspan fill="#c9d1d9">{esc(text)}</tspan>'
        elif who == "cont":
            t = f'<tspan fill="#c9d1d9" xml:space="preserve">     {esc(text)}</tspan>'
        else:
            t = f'<tspan fill="#7d8590" xml:space="preserve">     {esc(text)}</tspan>'
        lines.append(f'<text class="mono" x="40" y="{y}" font-size="20" xml:space="preserve">{t}</text>')
        y += lh
    body = f"""<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="#0d1117" stroke="#30363d"/>
<path d="M0.5 14.5a14 14 0 0 1 14-14h{W - 29}a14 14 0 0 1 14 14v31h-{W - 1}z" fill="#161b22"/>
<circle cx="30" cy="23" r="7" fill="#ff5f57"/><circle cx="54" cy="23" r="7" fill="#febc2e"/><circle cx="78" cy="23" r="7" fill="#28c840"/>
<text x="{W / 2}" y="29" font-size="18" fill="#8b949e" text-anchor="middle">MacBook Air M4 · 24 GB — DeepSeek-V4.1-Flash, greedy</text>
{chr(10).join(lines)}"""
    return svg(W, H, body, "Real chat output of DeepInfer-Mac on a MacBook Air M4 with 24 GB")


# ------------------------------------------------------------------------------ architecture

def arrow(x1, y1, x2, y2, kind=""):
    w, a = {"": ("wire", "ah"), "read": ("wire-read", "ah-read"), "spec": ("wire-spec", "ah-spec")}[kind]
    s = f'<line class="{w}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="2.5"/>'
    if x2 > x1:
        s += f'<path class="{a}" d="M{x2} {y2} l-11 -6 v12 z"/>'
    elif x2 < x1:
        s += f'<path class="{a}" d="M{x2} {y2} l11 -6 v12 z"/>'
    elif y2 > y1:
        s += f'<path class="{a}" d="M{x2} {y2} l-6 -11 h12 z"/>'
    else:
        s += f'<path class="{a}" d="M{x2} {y2} l6 11 h-12 z"/>'
    return s


def box(x, y, w, h, title, sub):
    s = f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" stroke-width="1.5"/>'
    s += f'<text class="t1" x="{x + w / 2}" y="{y + 34}" font-size="21" font-weight="600" text-anchor="middle">{title}</text>'
    for i, line in enumerate(sub):
        s += f'<text class="t2" x="{x + w / 2}" y="{y + 62 + i * 25}" font-size="17" text-anchor="middle">{line}</text>'
    return s


def architecture():
    """One decode layer, the speculative path, and where time and memory go."""
    W, H = 1280, 690
    b = [card(W, H),
         '<text class="t1" x="48" y="60" font-size="28" font-weight="700">One layer of a decode step (×40 per token)</text>',
         '<text class="t2" x="48" y="92" font-size="19">Dense weights stay in memory. Missing experts are read from the SSD; the next layer\'s likely experts follow them.</text>',
         '<rect class="gpu" x="40" y="120" width="780" height="340" rx="14" stroke-width="1.5" stroke-dasharray="6 5"/>',
         '<text class="accent" x="62" y="150" font-size="17" font-weight="700" letter-spacing="1">UNIFIED MEMORY · GPU</text>',
         '<rect class="ssdbox" x="990" y="120" width="250" height="340" rx="14" stroke-width="1.5" stroke-dasharray="6 5"/>',
         '<text class="accent2" x="1010" y="150" font-size="17" font-weight="700" letter-spacing="1">INTERNAL SSD</text>',
         box(62, 172, 220, 120, "Attention", ["sparse attention,", "indexer, mHC"]),
         box(312, 172, 200, 120, "Router", ["6 of 384 experts", "+ next layer's guess"]),
         box(542, 172, 256, 120, "Expert cache", ["LRU slots in MLX slabs", "guesses enter cold"]),
         box(542, 318, 256, 104, "gather_qmm", ["all experts of the layer,", "one quantized matmul"]),
         box(1008, 172, 214, 120, "Routed experts", ["40 × 384, 2–4-bit", f"{EXPERTS_GB} GB"]),
         box(1008, 318, 214, 104, "Engram tables", ["4-bit rows", f"{ENGRAM_GB} GB"]),
         arrow(282, 232, 310, 232), arrow(512, 232, 540, 232), arrow(670, 292, 670, 316),
         arrow(1006, 205, 800, 205, "read"), arrow(1006, 262, 800, 262, "spec"),
         '<text class="accent2" x="903" y="195" font-size="17" font-weight="600" text-anchor="middle">misses</text>',
         '<text class="accent3" x="903" y="292" font-size="17" font-weight="600" text-anchor="middle">next layer\'s</text>',
         '<text class="accent3" x="903" y="314" font-size="17" font-weight="600" text-anchor="middle">guesses, after</text>',
         '<text class="accent3" x="903" y="336" font-size="17" font-weight="600" text-anchor="middle">misses land</text>',
         '<line class="wire-read" x1="1115" y1="422" x2="1115" y2="444" stroke-width="2" stroke-dasharray="5 4"/>'
         '<line class="wire-read" x1="1115" y1="444" x2="172" y2="444" stroke-width="2" stroke-dasharray="5 4"/>'
         '<line class="wire-read" x1="172" y1="444" x2="172" y2="306" stroke-width="2" stroke-dasharray="5 4"/>'
         '<path class="ah-read" d="M172 294 l-6 12 h12 z"/>',
         '<text class="t2" x="330" y="434" font-size="16" text-anchor="middle">Engram rows (layers 1, 14)</text>']
    # time and memory bars at the 5.5 GB operating point
    y0, bx, bw = 520, 48, 540
    comp = TOKEN_S - SSD_WAIT_S
    b.append(f'<text class="t1" x="{bx}" y="{y0}" font-size="21" font-weight="700">One token: {TOKEN_S:.2f} s ({DECODE_TOKS:.2f} tok/s)</text>')
    ww = bw * SSD_WAIT_S / TOKEN_S
    b.append(f'<rect class="read" x="{bx}" y="{y0 + 18}" width="{ww - 1.5}" height="38" rx="4"/>')
    b.append(f'<rect class="ram" x="{bx + ww + 1.5}" y="{y0 + 18}" width="{bw - ww - 1.5}" height="38" rx="4"/>')
    b.append(f'<text class="t2" x="{bx}" y="{y0 + 86}" font-size="17">SSD wait {SSD_WAIT_S:.2f} s</text>')
    b.append(f'<text class="t2" x="{bx + bw}" y="{y0 + 86}" font-size="17" text-anchor="end">compute {comp:.2f} s</text>')
    b.append(f'<text class="t3" x="{bx}" y="{y0 + 116}" font-size="17">5.5 GB cache · per token {DEMAND_GB:.1f} GB needed, {SPEC_GB:.1f} GB read ahead</text>')
    mx_, mw = 680, 552
    b.append(f'<text class="t1" x="{mx_}" y="{y0}" font-size="21" font-weight="700">Where the 24 GB go</text>')
    parts = [("ram", RESIDENT_GB, "weights", f"{RESIDENT_GB:.1f}"), ("read", 5.5, "cache", "5.5"), ("disk", 24 - RESIDENT_GB - 5.5, "macOS, apps, KV", "≥9")]
    x = mx_
    for i, (cls, gb, lab, val) in enumerate(parts):
        w = mw * gb / 24
        b.append(f'<rect class="{cls}" x="{x + (1.5 if i else 0)}" y="{y0 + 18}" width="{w - 3}" height="38" rx="4"/>')
        b.append(f'<text class="t2" x="{x + 2}" y="{y0 + 86}" font-size="17">{lab} {val} GB</text>')
        x += w
    b.append(f'<text class="t3" x="{mx_}" y="{y0 + 116}" font-size="17">The cache sizes itself so that macOS keeps ≥ 9 GB.</text>')
    return svg(W, H, "\n".join(b), "How DeepInfer-Mac streams experts from the SSD, and where time and memory go")


# ------------------------------------------------------------------------------ speed vs. cache

def speed():
    """Decode speed against expert cache size, with and without idle-window speculation."""
    rows = sorted(SWEEP.values(), key=lambda r: r["cache"])
    W, H = 1280, 580
    px, py, pw, ph = 120, 120, 860, 330
    xmin, xmax, ymin, ymax = 2.25, 6.75, 0.6, 1.5
    X = lambda c: px + (c - xmin) / (xmax - xmin) * pw  # noqa: E731
    Y = lambda v: py + ph - (v - ymin) / (ymax - ymin) * ph  # noqa: E731
    b = [card(W, H),
         '<text class="t1" x="48" y="58" font-size="28" font-weight="700">Decode speed vs. expert cache size</text>',
         '<text class="t2" x="48" y="90" font-size="19">MacBook Air M4, 24 GB · both settings interleaved in one process, 3 prompts × 3 rounds per point</text>']
    for v in (0.6, 0.8, 1.0, 1.2, 1.4):
        b.append(f'<line class="grid" x1="{px}" y1="{Y(v)}" x2="{px + pw}" y2="{Y(v)}" stroke-width="1"/>'
                 f'<text class="t3" x="{px - 14}" y="{Y(v) + 6}" font-size="17" text-anchor="end">{v:.1f}</text>')
    for c in (2.5, 3.5, 4.5, 5.5, 6.5):
        b.append(f'<text class="t3" x="{X(c)}" y="{py + ph + 30}" font-size="17" text-anchor="middle">{c:g} GB</text>')
    b.append(f'<line class="axis" x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke-width="1.5"/>')
    b.append(f'<text class="t2" x="{px - 78}" y="{py + ph / 2}" font-size="17" transform="rotate(-90 {px - 78} {py + ph / 2})" text-anchor="middle">tokens/s</text>')
    b.append(f'<text class="t2" x="{px + pw / 2}" y="{py + ph + 62}" font-size="17" text-anchor="middle">expert cache size</text>')
    safe = X(6.0)
    b.append(f'<line class="axis" x1="{safe}" y1="{py}" x2="{safe}" y2="{py + ph}" stroke-width="1" stroke-dasharray="4 4"/>'
             f'<text class="t3" x="{safe - 8}" y="{py + ph - 12}" font-size="16" text-anchor="end">keeps 9 GB for macOS ←</text>')
    for key, cls, dcls, lab in (("tok_ov", "l-ov", "d-ov", "without speculation"), ("tok_iw", "l-iw", "d-iw", "with speculation")):
        pts = " ".join(f"{X(r['cache']):.1f},{Y(r[key]):.1f}" for r in rows)
        b.append(f'<polyline class="{cls}" points="{pts}" fill="none" stroke-width="3"/>')
        ci = "tok_ci_ov" if key == "tok_ov" else "tok_ci_iw"
        for r in rows:
            b.append(f'<line class="{cls}" x1="{X(r["cache"])}" y1="{Y(r[key] - r[ci])}" x2="{X(r["cache"])}" y2="{Y(r[key] + r[ci])}" stroke-width="1.5"/>')
            b.append(f'<circle class="{dcls} ring" cx="{X(r["cache"])}" cy="{Y(r[key])}" r="7" stroke-width="2.5"/>')
        last = rows[-1]
        b.append(f'<text class="t1" x="{X(last["cache"]) + 18}" y="{Y(last[key]) + 2}" font-size="19" font-weight="700">{last[key]:.2f}</text>')
        b.append(f'<text class="t2" x="{X(last["cache"]) + 18}" y="{Y(last[key]) + 24}" font-size="16">{lab}</text>')
    for r in rows:  # paired speed-up above each pair of points
        b.append(f'<text class="accent3" x="{X(r["cache"])}" y="{Y(r["tok_iw"] + r["tok_ci_iw"]) - 14}" font-size="17" font-weight="700" text-anchor="middle">+{(r["ratio"] - 1) * 100:.0f}%</text>')
    b.append(legend(px, py + ph + 102, [("d-ov", "without speculation"), ("d-iw", "idle-window speculation (default)")]))
    return svg(W, H, "\n".join(b), "Decode speed against expert cache size, with and without idle-window speculation")


# ------------------------------------------------------------------------------ timeline

def _union(iv):
    iv, out = sorted(iv), []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def timeline():
    """Four decode layers from real traces: GPU and SSD activity, without and with speculation."""
    d = json.loads((DATA / "trace_layers.json").read_text())
    W, H = 1280, 610
    px, pw = 150, 1080
    span = 100.0  # ms shown
    b = [card(W, H),
         '<text class="t1" x="48" y="58" font-size="28" font-weight="700">Filling the SSD\'s idle time</text>',
         '<text class="t2" x="48" y="90" font-size="19">Four decode layers from real traces (5 GB cache). Without speculation, the SSD waits while the GPU works.</text>']
    busy_pct = {"off": 62, "spec": 73}  # SSD busy share of the whole (median) traced step, paper/logs/trace_clean_*.json
    for pi, (key, title) in enumerate((("off", "Without speculation"), ("spec", "Idle-window speculation"))):
        ev = d[key]["events"]
        t0 = min(e["t0"] for e in ev if e["kind"] == "route") - 0.008
        ms = lambda t: (t - t0) * 1e3  # noqa: E731
        X = lambda t: px + min(max(t, 0), span) / span * pw  # noqa: E731
        y = 140 + pi * 190
        b.append(f'<text class="t1" x="48" y="{y + 4}" font-size="21" font-weight="600">{title}</text>')
        b.append(f'<text class="t2" x="{px + pw}" y="{y + 4}" font-size="18" text-anchor="end">SSD busy {busy_pct[key]}% of the step</text>')
        gy, sy = y + 22, y + 84
        b.append(f'<text class="t2" x="48" y="{gy + 29}" font-size="18">GPU</text><text class="t2" x="48" y="{sy + 29}" font-size="18">SSD</text>')
        b.append(f'<rect class="disk" x="{px}" y="{sy}" width="{pw}" height="44" rx="4" opacity="0.45"/>')
        route = {e["layer"]: e["t0"] for e in ev if e["kind"] == "route"}
        prev = None
        for L in d[key]["layers"]:
            es = [e for e in ev if e["layer"] == L]
            r = route[L]
            start = ms(prev) if prev is not None else 0.0
            b.append(f'<rect class="attn" x="{X(start)}" y="{gy}" width="{max(X(ms(r)) - X(start) - 2, 1)}" height="44" rx="4"/>')
            waits = [e for e in es if e["kind"] == "wait"]
            for g in [e for e in es if e["kind"] == "gpu"]:
                a, z = ms(g["t0"]), ms(g["t1"])
                for w in waits:
                    if w["t0"] >= g["t0"] - 1e-4 and w["t1"] <= g["t1"] + 1e-4 and w["t1"] > w["t0"]:
                        a = max(a, ms(w["t1"]))
                b.append(f'<rect class="ram" x="{X(a)}" y="{gy}" width="{max(X(z) - X(a), 3)}" height="44" rx="4"/>')
            prev = max(e.get("t1", e["t0"]) for e in es)
        reads = [e for e in ev if e["kind"] == "read"]
        for spec_flag, cls in ((False, "read"), (True, "spec")):
            iv = [(ms(e["t0"]), ms(e["t1"])) for e in reads if (e["t0"] < route[e["layer"]] - 1e-4) == spec_flag]
            for a, z in _union(iv):
                if a < span:
                    b.append(f'<rect class="{cls}" x="{X(a)}" y="{sy}" width="{max(X(min(z, span)) - X(a), 2)}" height="44" rx="4"/>')
    for t in range(0, 101, 20):
        b.append(f'<text class="t3" x="{px + t / span * pw}" y="538" font-size="16" text-anchor="middle">{t} ms</text>')
    b.append(legend(px, 582, [("attn", "attention + router"), ("ram", "expert compute"), ("read", "needed reads"), ("spec", "read-ahead")], gap=30))
    return svg(W, H, "\n".join(b), "Decode timeline with and without idle-window speculation")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, f in (("hero", hero), ("chat", terminal), ("architecture", architecture), ("speed", speed), ("timeline", timeline)):
        (OUT / f"{name}.svg").write_text(f(), encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")
