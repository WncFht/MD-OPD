#!/usr/bin/env python3
"""Extract the paper's vector-figure curves into tidy CSVs (M1 input B).

The arXiv/translated figure PDFs are matplotlib vector exports: curves are
stroked polylines, axes are calibrated by gridlines, and tick values come from
the embedded text labels. This script

  1. collects stroked paths (PyMuPDF ``get_drawings``),
  2. finds gray gridlines (color ~ (0.8, 0.8, 0.8)) and clusters them into
     panels (columns of vertical gridlines),
  3. assigns tick values from nearby text labels ('1/2' -> 0.5, ...),
  4. verifies log-scale by checking equal log spacing of y gridlines,
  5. maps every non-gray polyline into (step, value) coordinates,
  6. writes one CSV per (figure, panel, series color) into --out.

Usage:
    python plots/extract_paper_figures.py --pdf <fig.pdf> --out plots/paper_curves
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pymupdf

GRID_TOL = 0.04  # color tolerance for "gray gridline"


def is_gray(color) -> bool:
    if color is None:
        return False
    r, g, b = color
    return abs(r - 0.8) < GRID_TOL and abs(g - 0.8) < GRID_TOL and abs(b - 0.8) < GRID_TOL


def parse_label(text: str) -> float | None:
    text = text.strip().replace("¡", "-").replace("£", "x")
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    if m:
        return float(m.group(1)) / float(m.group(2))
    # scientific: merged text uses x for the times sign and - for the minus:
    # "2x10-2" -> 2e-2, "10-2" -> 1e-2
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*x\s*10(-?\d+)", text)
    if m:
        return float(m.group(1)) * 10.0 ** float(m.group(2))
    m = re.fullmatch(r"10(-?\d+)", text)
    if m:
        return 10.0 ** float(m.group(1))
    return None


def merge_label_words(words: list) -> str:
    return "".join(w[4] for w in words)


def flatten_points(drawing: dict) -> list[list[tuple[float, float]]]:
    """Flatten a drawing into *subpaths* (a moveto starts a new subpath)."""
    subpaths: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []

    def bezier(p0, p1, p2, p3, n=10):
        out = []
        for i in range(1, n + 1):
            t = i / n
            u = 1 - t
            x = u**3 * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t**3 * p3.x
            y = u**3 * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t**3 * p3.y
            out.append((x, y))
        return out

    def close_subpath():
        nonlocal cur
        if len(cur) >= 2:
            subpaths.append(dedupe(cur))
        cur = []

    for item in drawing["items"]:
        op = item[0]
        if op == "l":
            p1, p2 = item[1], item[2]
            if cur and (abs(cur[-1][0] - p1.x) > 1e-3 or abs(cur[-1][1] - p1.y) > 1e-3):
                close_subpath()
            if not cur:
                cur.append((p1.x, p1.y))
            cur.append((p2.x, p2.y))
        elif op == "c":
            p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
            if cur and (abs(cur[-1][0] - p1.x) > 1e-3 or abs(cur[-1][1] - p1.y) > 1e-3):
                close_subpath()
            if not cur:
                cur.append((p1.x, p1.y))
            cur.extend(bezier(p1, p2, p3, p4))
        elif op == "re":
            close_subpath()
            rect = item[1]
            subpaths.append(
                [
                    (rect.x0, rect.y0),
                    (rect.x1, rect.y0),
                    (rect.x1, rect.y1),
                    (rect.x0, rect.y1),
                ]
            )
    close_subpath()
    return subpaths


def dedupe(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if not out or abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out


def cluster_1d(values: list[float], tol: float = 2.0) -> list[float]:
    """Collapse near-duplicate coordinates (same gridline drawn per panel/row)."""
    vals = sorted(values)
    clusters: list[list[float]] = []
    for v in vals:
        if clusters and v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "paper_curves")
    parser.add_argument("--inspect", action="store_true", help="Print diagnostics only")
    args = parser.parse_args()

    import pandas as pd

    doc = pymupdf.open(args.pdf)
    page = doc[0]
    drawings = page.get_drawings()
    words = page.get_text("words")  # x0, y0, x1, y1, text, ...

    # ---- gridlines ---------------------------------------------------------
    v_grid: list[tuple[float, float, float]] = []  # (x, y0, y1)
    h_grid: list[tuple[float, float, float]] = []  # (y, x0, x1)
    for d in drawings:
        if d["type"] != "s" or not is_gray(d.get("color")):
            continue
        for sub in flatten_points(d):
            for (x0, y0), (x1, y1) in zip(sub, sub[1:], strict=False):
                if abs(x0 - x1) < 0.5 and abs(y0 - y1) > 20:
                    v_grid.append((x0, min(y0, y1), max(y0, y1)))
                elif abs(y0 - y1) < 0.5 and abs(x0 - x1) > 20:
                    h_grid.append((y0, min(x0, x1), max(x0, x1)))

    # cluster gridline coordinates (per panel)
    v_xs = cluster_1d([x for x, _, _ in v_grid], tol=3.0)
    h_ys_all = cluster_1d([y for y, _, _ in h_grid], tol=3.0)

    # panels: black spine lines form the axes frames (2 vertical + 2 horizontal
    # strokes of the same extent per panel)
    black_v: list[tuple[float, float, float]] = []  # (x, y0, y1)
    black_h: list[tuple[float, float, float]] = []  # (y, x0, x1)
    for d in drawings:
        if d["type"] != "s" or not d.get("color") or max(d["color"]) >= 0.1:
            continue
        for item in d["items"]:
            if item[0] != "l":
                continue
            x0, y0, x1, y1 = item[1].x, item[1].y, item[2].x, item[2].y
            if abs(x0 - x1) < 0.5 and abs(y0 - y1) > 50:
                black_v.append((x0, min(y0, y1), max(y0, y1)))
            elif abs(y0 - y1) < 0.5 and abs(x0 - x1) > 50:
                black_h.append((y0, min(x0, x1), max(x0, x1)))
    panels: list[dict] = []
    for x, y0, y1 in black_v:
        # a left spine must be the *start* of its top/bottom horizontal partners
        partner_top = [h for h in black_h if abs(h[0] - y0) < 1 and abs(min(h[1], h[2]) - x) < 1]
        partner_bot = [h for h in black_h if abs(h[0] - y1) < 1 and abs(min(h[1], h[2]) - x) < 1]
        if not partner_top or not partner_bot:
            continue
        x_right = max(max(h[1], h[2]) for h in partner_top + partner_bot)
        entry = {"x0": x, "x1": x_right, "top": y0, "bottom": y1}
        if panels and abs(entry["x0"] - panels[-1]["x1"]) < 1 and abs(entry["top"] - panels[-1]["top"]) < 1:
            panels[-1]["x1"] = entry["x1"]
        else:
            panels.append(entry)
    panels = [p for p in panels if p["x1"] - p["x0"] > 100]
    for p in panels:
        p["xs"] = [x for x in v_xs if p["x0"] - 0.5 <= x <= p["x1"] + 0.5]

    if args.inspect:
        print(f"page={page.rect}")
        print(f"vertical gridline xs: {[round(x, 1) for x in v_xs]}")
        print(f"horizontal gridline ys: {[round(y, 1) for y in h_ys_all]}")
        for i, p in enumerate(panels):
            print(f"panel {i}: x0={p['x0']:.1f} x1={p['x1']:.1f} gridline xs={[round(x, 1) for x in p['xs']]}")
        colors = defaultdict(int)
        for d in drawings:
            if d["type"] == "s" and not is_gray(d.get("color")):
                colors[tuple(round(c, 3) for c in d["color"])] += len(d["items"])
        print("non-gray stroke colors (color: n_segments):")
        for c, n in sorted(colors.items(), key=lambda kv: -kv[1]):
            print(f"  {c}: {n}")
        return 0

    if not panels:
        raise SystemExit("no panels found")

    # ---- assign tick values from labels -------------------------------------
    def word_center(w) -> tuple[float, float]:
        return (w[0] + w[2]) / 2, (w[1] + w[3]) / 2

    series_out: list[pd.DataFrame] = []
    for pi, p in enumerate(panels):
        left, right = p["x0"], p["x1"]
        top, bottom = p["top"], p["bottom"]
        # x labels: below panel bottom
        x_labels: dict[float, float] = {}  # gridline x -> value
        for w in words:
            cx, cy = word_center(w)
            val = parse_label(w[4])
            if val is None or not (bottom - 5 < cy < bottom + 25):
                continue
            if not (left - 10 <= cx <= right + 10):
                continue
            gx = min(p["xs"], key=lambda gx: abs(gx - cx))
            if abs(gx - cx) < 12:
                x_labels[gx] = val
        # y labels: left of panel; group multi-word scientific labels by row
        h_ys = sorted({y for y, x0, x1 in h_grid if left - 2 <= x0 and x1 <= right + 2 and top - 5 <= y <= bottom + 5})
        cand = [w for w in words if left - 170 < (w[0] + w[2]) / 2 < left - 2]
        y_labels: dict[float, float] = {}
        for gy in h_ys:  # gridline-anchored: merge all label words near this line
            ws = sorted(
                (w for w in cand if abs((w[1] + w[3]) / 2 - gy) < 25),
                key=lambda w: w[0],
            )
            if not ws:
                continue
            val = parse_label(merge_label_words(ws))
            if val is not None:
                y_labels[gy] = val
        if len(x_labels) < 2 or len(y_labels) < 2:
            print(f"[warn] panel {pi}: calibration weak (x_labels={x_labels}, y_labels={y_labels})")
            continue

        # calibration maps
        xs = sorted(x_labels)
        ys = sorted(y_labels, reverse=True)  # top -> bottom
        vals_desc = [y_labels[y] for y in ys]
        # linear-vs-log axis: fit both through the tick (y-position, value) pairs
        import math

        def _r2(ypos: list[float], vals: list[float], log: bool) -> float:
            t = [math.log(v) if log else v for v in vals]
            n = len(t)
            mt, my = sum(t) / n, sum(ypos) / n
            sxx = sum((a - mt) ** 2 for a in t)
            sxy = sum((a - mt) * (b - my) for a, b in zip(t, ypos, strict=False))
            slope = sxy / sxx if sxx else 0.0
            ss_res = sum((b - (my + slope * (a - mt))) ** 2 for a, b in zip(t, ypos, strict=False))
            ss_tot = sum((b - my) ** 2 for b in ypos)
            return 1 - ss_res / ss_tot if ss_tot else 0.0

        log_scale = min(vals_desc) > 0 and _r2(ys, vals_desc, True) > _r2(ys, vals_desc, False)

        def map_x(px: float, xs=xs, x_labels=x_labels) -> float:
            import bisect

            pos = bisect.bisect_left(xs, px)
            if pos == 0:
                x0, x1 = xs[0], xs[1]
            elif pos >= len(xs):
                x0, x1 = xs[-2], xs[-1]
            else:
                x0, x1 = xs[pos - 1], xs[pos]
            v0, v1 = x_labels[x0], x_labels[x1]
            return v0 + (px - x0) * (v1 - v0) / (x1 - x0)

        def map_y(py: float, y_labels=y_labels, log_scale=log_scale) -> float:
            import bisect

            gys = sorted(y_labels)
            pos = bisect.bisect_left(gys, py)
            if pos == 0:
                y0, y1 = gys[0], gys[1]
            elif pos >= len(gys):
                y0, y1 = gys[-2], gys[-1]
            else:
                y0, y1 = gys[pos - 1], gys[pos]
            v0, v1 = y_labels[y0], y_labels[y1]
            if log_scale:
                import math

                l0, l1 = math.log(v0), math.log(v1)
                return math.exp(l0 + (py - y0) * (l1 - l0) / (y1 - y0))
            return v0 + (py - y0) * (v1 - v0) / (y1 - y0)

        # ---- curves: non-gray strokes inside the panel ----------------------
        # each subpath is kept separate: one PDF path object may pack several
        # curves of the same color (seen in the collapsed-axis figure)
        colors = defaultdict(list)
        for d in drawings:
            if d["type"] != "s" or is_gray(d.get("color")) or d.get("color") is None:
                continue
            col = tuple(round(c, 3) for c in d["color"])
            for sub in flatten_points(d):
                if len(sub) < 5:
                    continue
                inside = [pt for pt in sub if left - 2 <= pt[0] <= right + 2 and top - 2 <= pt[1] <= bottom + 2]
                if len(inside) < 5:
                    continue
                colors[col].append(inside)

        n_series = 0
        for col, branches in sorted(colors.items()):
            for bi, pts in enumerate(branches):
                df = pd.DataFrame(
                    {
                        "step": [map_x(x) for x, _ in pts],
                        "value": [map_y(y) for _, y in pts],
                        "panel": pi,
                    }
                ).sort_values("step")
                name = "-".join(f"{int(c * 255):02x}" for c in col)
                suffix = f"__s{bi}" if len(branches) > 1 else ""
                out = args.out / f"{args.pdf.stem}__panel{pi}__{name}{suffix}.csv"
                args.out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                series_out.append(df)
                n_series += 1
                print(
                    f"[extract] {out.name}: {len(df)} pts, step {df['step'].min():.0f}..{df['step'].max():.0f}, "
                    f"value {df['value'].min():.3g}..{df['value'].max():.3g} (log_scale={log_scale})"
                )
        if n_series == 0:
            print(f"[warn] panel {pi}: no curve strokes found inside bounds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
