"""Draw the README figures (docs/assets/*.svg) from the measured numbers in the README.

The SVGs follow the viewer's light/dark setting (prefers-color-scheme), use system fonts only, and
are plain text so they diff and stay small. Run: python scripts/make_figures.py
"""
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"

# measured / on-disk numbers (README: weight format, performance)
RESIDENT_GB = 7.7 + 0.7          # dense mxfp8 + output head
TOTAL_GB = 338                   # du of the converted checkpoint, GiB
PER_TOKEN_READ_GB = 1.8          # expert bytes per token at a 49% hit rate
DECODE_TOKS, TOKEN_S, SSD_WAIT_S = 1.25, 0.80, 0.55

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'PingFang SC', 'Noto Sans Mono CJK SC', monospace"

THEME = """
  .card { fill: #fcfcfb; stroke: #e4e3df; }
  .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #7a7974; }
  .ram { fill: #2a78d6; } .read { fill: #eb6834; } .disk { fill: #dcdad3; }
  .gpu { fill: #e9f1fc; stroke: #2a78d6; } .ssdbox { fill: #fdf0ea; stroke: #eb6834; }
  .box { fill: #ffffff; stroke: #c9c7bf; } .wire { stroke: #8a8984; } .ah { fill: #8a8984; }
  .wire-read { stroke: #eb6834; } .ah-read { fill: #eb6834; }
  .accent { fill: #1c5cab; } .accent2 { fill: #c4501f; }
  @media (prefers-color-scheme: dark) {
    .card { fill: #161b22; stroke: #30363d; }
    .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .t3 { fill: #94938c; }
    .ram { fill: #3987e5; } .read { fill: #d95926; } .disk { fill: #3a3a37; }
    .gpu { fill: #14243a; stroke: #3987e5; } .ssdbox { fill: #2e1c14; stroke: #d95926; }
    .box { fill: #0d1117; stroke: #484f58; } .wire { stroke: #8b949e; } .ah { fill: #8b949e; }
    .wire-read { stroke: #e0703f; } .ah-read { fill: #e0703f; }
    .accent { fill: #86b6ef; } .accent2 { fill: #f08a5d; }
  }
"""


def svg(w, h, body, title, text_scale=1.0):
    """text_scale enlarges text below 20 px: GitHub shows these 1280-px figures at ~65% width."""
    if text_scale != 1.0:
        body = re.sub(r'font-size="(\d+)"', lambda m: f'font-size="{round(int(m.group(1)) * text_scale) if int(m.group(1)) < 20 else m.group(1)}"', body)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{title}">\n<title>{title}</title>\n<style>{THEME}  text {{ font-family: {SANS}; }}\n'
            f'  .mono {{ font-family: {MONO}; }}\n</style>\n{body}\n</svg>\n')


def hero():
    """One square per GB of converted weights: what stays in memory, what one token reads, the rest."""
    W, H = 1280, 480
    cols, cell, gap = 26, 17, 4
    total = n = TOTAL_GB
    n_ram, n_read = round(RESIDENT_GB), round(PER_TOKEN_READ_GB)
    gx, gy = 680, 92
    sq = []
    for i in range(n):
        r, c = divmod(i, cols)
        cls = "ram" if i < n_ram else "read" if i < n_ram + n_read else "disk"
        sq.append(f'<rect class="{cls}" x="{gx + c * (cell + gap)}" y="{gy + r * (cell + gap)}" width="{cell}" height="{cell}" rx="3"/>')
    rows = -(-n // cols)
    gb = gy + rows * (cell + gap) + 18
    legend = (
        f'<rect class="ram" x="{gx}" y="{gb}" width="12" height="12" rx="3"/>'
        f'<text class="t2" x="{gx + 18}" y="{gb + 11}" font-size="14">{RESIDENT_GB:.1f} GB in memory</text>'
        f'<rect class="read" x="{gx + 190}" y="{gb}" width="12" height="12" rx="3"/>'
        f'<text class="t2" x="{gx + 210}" y="{gb + 11}" font-size="14">~{PER_TOKEN_READ_GB} GB read per token</text>'
        f'<rect class="disk" x="{gx + 432}" y="{gb}" width="12" height="12" rx="3"/>'
        f'<text class="t2" x="{gx + 452}" y="{gb + 11}" font-size="14">on the SSD</text>'
    )
    stats = [(f"{DECODE_TOKS}", "tok/s decode"), ("96.7%", "match with the API"), ("16K", "token retrieval")]
    st = []
    for j, (v, lab) in enumerate(stats):
        x = 64 + j * 200
        st.append(f'<text class="accent" x="{x}" y="412" font-size="34" font-weight="700">{v}</text>'
                  f'<text class="t2" x="{x}" y="440" font-size="13">{lab}</text>')
    body = f"""<rect class="card" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="16"/>
<text class="t3" x="64" y="96" font-size="15" font-weight="600" letter-spacing="1.5">DEEPSEEK-V4.1-FLASH · MLX · APPLE SILICON</text>
<text class="t1" x="64" y="160" font-size="56" font-weight="700" letter-spacing="-1">DeepInfer-Mac</text>
<text class="t1" x="64" y="218" font-size="27" font-weight="600">A 754B-parameter model</text>
<text class="t1" x="64" y="254" font-size="27" font-weight="600">on a 24 GB MacBook Air.</text>
<text class="t2" x="64" y="298" font-size="16">{round(total)} GB of weights; {RESIDENT_GB:.1f} GB stay in memory.</text>
<text class="t2" x="64" y="326" font-size="16">Each token streams only the 6 of 384 experts</text>
<text class="t2" x="64" y="354" font-size="16">each layer picks, straight from the SSD.</text>
{''.join(st)}
<text class="t3" x="{gx}" y="76" font-size="13">each square = 1 GB of converted weights</text>
{''.join(sq)}
{legend}"""
    return svg(W, H, body, f"DeepInfer-Mac: {round(total)} GB of DeepSeek-V4.1-Flash weights, {RESIDENT_GB:.1f} GB in memory", text_scale=1.3)


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


def arrow(x1, y1, x2, y2, read=False, label=None, lx=None, ly=None):
    w, a = ("wire-read", "ah-read") if read else ("wire", "ah")
    s = f'<line class="{w}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="2"/>'
    if x2 > x1:
        s += f'<path class="{a}" d="M{x2} {y2} l-9 -5 v10 z"/>'
    elif y2 > y1:
        s += f'<path class="{a}" d="M{x2} {y2} l-5 -9 h10 z"/>'
    else:
        s += f'<path class="{a}" d="M{x2} {y2} l5 9 h-10 z"/>'
    if label:
        s += f'<text class="t2" x="{lx}" y="{ly}" font-size="13" text-anchor="middle">{label}</text>'
    return s


def box(x, y, w, h, title, sub, cls="box"):
    s = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" stroke-width="1.5"/>'
    s += f'<text class="t1" x="{x + w / 2}" y="{y + 30}" font-size="16" font-weight="600" text-anchor="middle">{title}</text>'
    for i, line in enumerate(sub):
        s += f'<text class="t2" x="{x + w / 2}" y="{y + 58 + i * 22}" font-size="13" text-anchor="middle">{line}</text>'
    return s


def architecture():
    """One decode step through one layer, and where the 24 GB go."""
    W, H = 1280, 610
    b = [f'<rect class="card" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="16"/>',
         '<text class="t1" x="48" y="58" font-size="22" font-weight="700">One token, one layer — ×40 layers per token</text>',
         '<text class="t2" x="48" y="84" font-size="14">Dense weights stay in memory; an expert is read from the SSD only when it is not cached.</text>']
    # memory region and SSD region
    b.append('<rect class="gpu" x="40" y="110" width="790" height="330" rx="14" stroke-width="1.5" stroke-dasharray="6 5"/>')
    b.append('<text class="accent" x="60" y="136" font-size="13" font-weight="700" letter-spacing="1">UNIFIED MEMORY · GPU</text>')
    b.append('<rect class="ssdbox" x="980" y="110" width="260" height="330" rx="14" stroke-width="1.5" stroke-dasharray="6 5"/>')
    b.append('<text class="accent2" x="1000" y="136" font-size="13" font-weight="700" letter-spacing="1">INTERNAL SSD</text>')
    b.append(box(64, 160, 220, 110, "Attention", ["sparse attention", "+ indexer, mHC, Engram", "mxfp8, bit-exact"]))
    b.append(box(320, 160, 200, 110, "Router", ["6 of 384 experts", "+ shared expert"]))
    b.append(box(556, 160, 250, 110, "Expert cache", ["LRU slots in MLX slabs", "~49% hit rate"]))
    b.append(box(556, 300, 250, 90, "gather_qmm", ["6 experts, one fused", "matmul per projection"]))
    b.append(box(1000, 160, 220, 110, "Routed experts", ["40 × 384, 2/3/4-bit", "213 GB"]))
    b.append(box(1000, 290, 220, 100, "Engram tables", ["384M rows × 2, 4-bit", "115 GB"]))
    b.append(arrow(284, 215, 318, 215))
    b.append(arrow(520, 215, 554, 215))
    b.append(arrow(681, 270, 681, 298))
    b.append(arrow(998, 215, 808, 215, read=True))
    b.append('<text class="accent2" x="903" y="203" font-size="13" font-weight="600" text-anchor="middle">on a miss</text>')
    b.append('<text class="t2" x="903" y="242" font-size="12" text-anchor="middle">one preadv,</text>')
    b.append('<text class="t3" x="903" y="260" font-size="12" text-anchor="middle">zero-copy</text>')
    b.append('<line class="wire-read" x1="1110" y1="390" x2="1110" y2="418" stroke-width="2" stroke-dasharray="4 4"/>'
             '<line class="wire-read" x1="1110" y1="418" x2="174" y2="418" stroke-width="2" stroke-dasharray="4 4"/>'
             '<line class="wire-read" x1="174" y1="418" x2="174" y2="282" stroke-width="2" stroke-dasharray="4 4"/>'
             '<path class="ah-read" d="M174 272 l-5 10 h10 z"/>'
             '<text class="t2" x="380" y="408" font-size="12" text-anchor="middle">Engram rows (layers 1, 14)</text>')
    # per-token time and memory budget
    y0, bw, bx = 484, 520, 48
    b.append(f'<text class="t1" x="{bx}" y="{y0}" font-size="16" font-weight="700">Where a token\'s {TOKEN_S:.2f} s goes</text>')
    wait_w = bw * SSD_WAIT_S / TOKEN_S
    b.append(f'<rect class="read" x="{bx}" y="{y0 + 18}" width="{wait_w - 1}" height="34" rx="4"/>')
    b.append(f'<rect class="ram" x="{bx + wait_w + 1}" y="{y0 + 18}" width="{bw - wait_w - 1}" height="34" rx="4"/>')
    b.append(f'<text class="t2" x="{bx}" y="{y0 + 76}" font-size="13">SSD wait {SSD_WAIT_S:.2f} s (~{PER_TOKEN_READ_GB} GB)</text>')
    b.append(f'<text class="t2" x="{bx + bw}" y="{y0 + 76}" font-size="13" text-anchor="end">compute {TOKEN_S - SSD_WAIT_S:.2f} s</text>')
    b.append(f'<text class="t3" x="{bx}" y="{y0 + 102}" font-size="13">SSD-bound: a bigger cache is the lever.</text>')
    mx_, mw = 680, 552
    b.append(f'<text class="t1" x="{mx_}" y="{y0}" font-size="16" font-weight="700">Where the 24 GB go</text>')
    parts = [("ram", RESIDENT_GB, f"weights {RESIDENT_GB:.1f} GB"), ("read", 24 - RESIDENT_GB - 9, "cache ≤6.6 GB"), ("disk", 9, "macOS ≥9 GB")]
    x = mx_
    for i, (cls, gb, lab) in enumerate(parts):
        w = mw * gb / 24
        b.append(f'<rect class="{cls}" x="{x + (1 if i else 0)}" y="{y0 + 18}" width="{w - 2}" height="34" rx="4"/>')
        b.append(f'<text class="t2" x="{x + 2}" y="{y0 + 76}" font-size="13">{lab}</text>')
        x += w
    b.append(f'<text class="t3" x="{mx_}" y="{y0 + 102}" font-size="13">The cache sizes itself at start-up.</text>')
    return svg(W, H, "\n".join(b), "How DeepInfer-Mac streams experts from the SSD, and where time and memory go", text_scale=1.25)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, f in (("hero", hero), ("chat", terminal), ("architecture", architecture)):
        (OUT / f"{name}.svg").write_text(f(), encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")
