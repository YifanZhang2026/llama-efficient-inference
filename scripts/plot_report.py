"""
Draw the figures in docs/REPORT.md as standalone SVG (no plotting library needed).

    python scripts/plot_report.py      # writes docs/fig/*.svg from results/*.csv
"""
import csv
import os
from xml.sax.saxutils import escape

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "fig")

# Light / dark tokens; the SVG switches with the viewer's color scheme.
STYLE = """
<style>
  .bg { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }
  .grid { stroke: #e4e3df; } .axis { stroke: #a8a7a0; }
  .accent { stroke: #2a78d6; fill: #2a78d6; } .accent-t { fill: #2a78d6; }
  .muted { stroke: #9e9d96; fill: #9e9d96; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }
    .grid { stroke: #383835; } .axis { stroke: #6b6a64; }
    .accent { stroke: #3987e5; fill: #3987e5; } .accent-t { fill: #3987e5; }
    .muted { stroke: #8a8983; fill: #8a8983; }
  }
  text { font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
</style>"""


def line_chart(path, title, subtitle, series, x_ticks, y_ticks, y_fmt, x_label,
               note=None, W=760, H=360):
    """series: [(name, [(x, y), ...], accent: bool, opacity)], drawn in list order."""
    left, right, y0, bot = 60, 560, 86, 310
    xs = [x for _, pts, _, _ in series for x, _ in pts]
    xmin, xmax = min(xs), max(xs)
    ylo, yhi = y_ticks[0], y_ticks[-1]
    X = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    Y = lambda y: bot - (y - ylo) / (yhi - ylo) * (bot - y0)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'role="img" aria-label="{escape(title)}" font-size="12">', STYLE,
         f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="t1" x="{left}" y="30" font-size="16">{escape(title)}</text>',
         f'<text class="t2" x="{left}" y="52">{escape(subtitle)}</text>']
    for v in y_ticks:
        cls = "axis" if v == 0 else "grid"
        o.append(f'<line class="{cls}" x1="{left}" x2="{right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="t2" x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{y_fmt(v)}</text>')
    for v in x_ticks:
        o.append(f'<text class="t2" x="{X(v):.1f}" y="{bot + 20}" text-anchor="middle">{v}</text>')
    o.append(f'<text class="t2" x="{right}" y="{bot + 40}" text-anchor="end">{escape(x_label)}</text>')

    labels = []
    for name, pts, accent, op in series:
        cls = "accent" if accent else "muted"
        d = " ".join(f"{'L' if i else 'M'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        o.append(f'<path class="{cls}" d="{d}" fill="none" stroke-width="{2.5 if accent else 2}" '
                 f'stroke-opacity="{op}"/>')
        for x, y in pts:
            o.append(f'<circle class="{cls}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill-opacity="{op}">'
                     f'<title>{escape(name)}, {x}: {y_fmt(y, 2)}</title></circle>')
        labels.append([Y(pts[-1][1]), f"{name}  {y_fmt(pts[-1][1], 2)}", accent])
    # end labels, pushed apart so they never overlap
    labels.sort()
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 16)
    for y, text, accent in labels:
        o.append(f'<text class="{"accent-t" if accent else "t2"}" x="{right + 10}" y="{y + 4:.1f}">{escape(text)}</text>')
    if note:
        nx, ny, text = note
        o.append(f'<text class="t1" x="{X(nx) + 6:.1f}" y="{Y(ny) - 12:.1f}">{escape(text)}</text>')
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


def read(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUT, exist_ok=True)

    # E7b: per-layer cosine to dense, teacher-forced FFN output vs free-running residual stream
    rows = sorted(read("slice_scale.csv"), key=lambda r: int(r["layer"]))
    tf = [(int(r["layer"]), float(r["c_act"])) for r in rows]
    fr = [(int(r["layer"]), float(r["c_h"])) for r in rows]
    l7 = dict(fr)[7]
    line_chart(os.path.join(OUT, "e7b_slice_cos.svg"),
               "自己跑时，残差流与 dense 的余弦到第 7 层已降到约 0",
               "未训练 slice MoE（top-2 of 4），5711 个 token 平均；teacher-forced = 每层 FFN 输入取自 dense",
               [("teacher-forced（单层 FFN）", tf, False, 1.0), ("free-running（残差流）", fr, True, 1.0)],
               x_ticks=[0, 3, 6, 9, 12, 15], y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
               y_fmt=lambda v, n=2: f"{v:.{n}f}", x_label="层",
               note=(7, l7, f"第 7 层：{l7:.3f}"))

    # E8c: fraction of SGD updates lost to rounding, per step
    last = {}
    for r in read("update_rounding.csv"):
        if r["group"] != "lora":  # first run pooled LoRA A and B; superseded
            last[(r["config"], r["group"], int(r["step"]))] = float(r["lost_pct"])
    def pts(config, group):
        return sorted((s, v) for (c, g, s), v in last.items() if c == config and g == group)
    line_chart(os.path.join(OUT, "e8c_update_lost.svg"),
               "放开 expert 训练时，expert 权重每步 99.95% 以上的更新被舍入掉",
               "|lr×(0.9·buf+grad)| 小于 ulp(w)/2 的元素比例；SGD momentum 0.9，clip 1.0",
               [("slice · router", pts("slice", "router"), False, 0.45),
                ("LoRA · lora_A（fp32）", pts("lora", "lora_A"), False, 0.7),
                ("unfreeze · router", pts("slice-unfreeze", "router"), False, 1.0),
                ("unfreeze · expert", pts("slice-unfreeze", "expert"), True, 1.0)],
               x_ticks=[1, 5, 10, 15, 20], y_ticks=[40, 60, 80, 100],
               y_fmt=lambda v, n=0: f"{v:.{n}f}%", x_label="训练步")


if __name__ == "__main__":
    main()
