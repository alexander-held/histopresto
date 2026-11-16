"""Write the histopresto logo as SVG: `pixi run logo` (see --help for the icon and dark variants)."""

import argparse
import math
import pathlib

NAVY = "#0b2a6b"
LIGHT = "#e9effb"  # replaces navy on dark backgrounds
CYAN = "#19b5f1"
ORANGE = "#f5841f"
FONT = "'Avenir Next', Nunito, Montserrat, 'Helvetica Neue', Arial, sans-serif"

BASELINE = 180
SHOULDER = 90  # top of the right leg of the "h", the name's ascenders align with it
BIN_WIDTH = 28
# the "h" as three histogram bins with a gap of 4 between them: x, top, bottom
BINS = [(100, 20, BASELINE), (132, SHOULDER, SHOULDER + BIN_WIDTH), (164, SHOULDER, BASELINE)]
STROKE = 9  # fit curve and speed lines
FIT = [(70, 176), (108, 64), (215, 18)]  # fitted curve (quadratic Bezier), ends in the data point
DOT = (220, 16, 14)  # data point: cx, cy, r
SPEED_LINES = [(30, 80, 122), (0, 70, 140), (22, 62, 158)]  # x1, x2, y, they end at the curve
GAP = 5  # clear space where the curve crosses the stem

ASCENDER = 0.76  # height of "h" over font size, for the first font in FONT
TEXT_X = 250
TEXT_WIDTH = 5.09  # width of the name over font size


def fit_offset(distance, steps=200):
    """Points along the fit curve, shifted sideways (upwards for positive distances)."""
    (x0, y0), (x1, y1), (x2, y2) = FIT
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * x0 + 2 * t * (1 - t) * x1 + t**2 * x2
        y = (1 - t) ** 2 * y0 + 2 * t * (1 - t) * y1 + t**2 * y2
        dx = (1 - t) * (x1 - x0) + t * (x2 - x1)
        dy = (1 - t) * (y1 - y0) + t * (y2 - y1)
        norm = math.hypot(dx, dy)
        points.append((x + distance * dy / norm, y - distance * dx / norm))
    return points


def stem():
    """The first bin in two pieces, leaving a gap around the fit curve."""
    x, top, bottom = BINS[0]
    pieces = []
    for y_end, distance in [(top, STROKE / 2 + GAP), (bottom, -STROKE / 2 - GAP)]:
        points = fit_offset(distance)
        edge = []
        for k in range(BIN_WIDTH, -1, -2):  # right to left, x increases along the curve
            (xa, ya), (xb, yb) = next(
                pair for pair in zip(points, points[1:]) if pair[0][0] <= x + k <= pair[1][0]
            )
            y = ya + (yb - ya) * (x + k - xa) / (xb - xa)
            edge.append(f"{x + k},{y:.2f}")
        pieces.append(f"M{x},{y_end} h{BIN_WIDTH} L{' '.join(edge)} z")
    return " ".join(pieces)


def mark(ink):
    cx, cy, r = DOT
    (x0, y0), (x1, y1), (x2, y2) = FIT
    bins = "\n".join(
        f'    <rect x="{x}" y="{top}" width="{BIN_WIDTH}" height="{bottom - top}"/>'
        for x, top, bottom in BINS[1:]
    )
    fades = "\n".join(
        f"""    <linearGradient id="fade{i}" gradientUnits="userSpaceOnUse" x1="{xa}" x2="{xb}" y1="0" y2="0">
      <stop offset="0" stop-color="{CYAN}" stop-opacity="0"/>
      <stop offset="1" stop-color="{CYAN}"/>
    </linearGradient>"""
        for i, (xa, xb, _) in enumerate(SPEED_LINES)
    )
    lines = "\n".join(
        f'    <line x1="{xa}" x2="{xb}" y1="{y}" y2="{y}" stroke="url(#fade{i})"/>'
        for i, (xa, xb, y) in enumerate(SPEED_LINES)
    )
    return f"""  <defs>
{fades}
  </defs>
  <g fill="none" stroke-width="{STROKE}" stroke-linecap="round">
{lines}
    <path d="M{x0},{y0} Q{x1},{y1} {x2},{y2}" stroke="{CYAN}"/>
  </g>
  <g fill="{ink}">
    <path d="{stem()}"/>
{bins}
  </g>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{ORANGE}"/>"""


def logo(mark_only=False, dark=False):
    ink = LIGHT if dark else NAVY
    if mark_only:  # square, with padding, for icons
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="-12 -38 258 258" role="img" aria-label="histopresto">
{mark(ink)}
</svg>
"""
    size = round((BASELINE - SHOULDER) / ASCENDER)
    width = round(size * TEXT_WIDTH)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -8 {TEXT_X + width + 20} 230" role="img" aria-label="histopresto">
{mark(ink)}
  <text x="{TEXT_X}" y="{BASELINE}" font-family="{FONT}" font-size="{size}" font-weight="500"
        textLength="{width}" fill="{ink}">histopresto</text>
</svg>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("logo.svg"))
    parser.add_argument("--mark-only", action="store_true", help="square icon without the name")
    parser.add_argument("--dark", action="store_true", help="for dark backgrounds")
    args = parser.parse_args()
    args.output.write_text(logo(args.mark_only, args.dark))
    print(f"wrote {args.output}")
