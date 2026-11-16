// Small SVG plotting helper: a figure holds several axes, marks are given in data coordinates.
// Colours are given as CSS values, so figures can follow custom properties of the page (the names used are listed in js/viz/README.md).

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const r2 = (v) => Math.round(v * 100) / 100;

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const plain = (s) => String(s).replace(/[_^]\{([^}]*)\}/g, "$1");
let measureCtx = null;
function textWidth(str, size, bold) {
  if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
  if (!measureCtx) return plain(str).length * size * 0.56;
  measureCtx.font = `${bold ? "600 " : ""}${size}px ${FONT}`;
  return measureCtx.measureText(plain(str)).width;
}
// font size at which the text fits into the given width
const fitted = (str, size, maxWidth, bold, min) => {
  const width = textWidth(str, size, bold);
  return width > maxWidth ? Math.max(min, (size * maxWidth) / width) : size;
};
// variables in italics: greek letters, the L of "ln L", and letters carrying a tilde
const math = (s) =>
  esc(s)
    .replace(/([α-ω]+|(?<=ln )L(?![a-z]))/g, '<tspan font-style="italic">$1</tspan>')
    .replace(/(\w)\u0303/g, '<tspan font-style="italic">$1</tspan><tspan dx="-0.56em">˜</tspan><tspan dx="0.12em"></tspan>');

// text with _{subscripts} and ^{superscripts} as tspans
function rich(str, size) {
  const parts = String(str).split(/([_^]\{[^}]*\})/);
  let out = "";
  let offset = 0; // current vertical offset in px
  for (const part of parts) {
    if (!part) continue;
    const m = part.match(/^([_^])\{([^}]*)\}$/);
    const target = m ? (m[1] === "_" ? 0.28 * size : -0.4 * size) : 0;
    const dy = target - offset;
    offset = target;
    out += `<tspan${dy ? ` dy="${r2(dy)}"` : ""}${m ? ` font-size="${r2(0.74 * size)}"` : ""}>${math(m ? m[2] : part)}</tspan>`;
  }
  return out;
}

function niceTicks(lo, hi, target) {
  const raw = ((hi - lo) / target) * (1 - 1e-9);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const steps = [0.5, 1, 2, 5, 10].map((m) => m * mag);
  const ticksFor = (step) => {
    const ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return ticks;
  };
  // the first nice step that is wide enough, one finer if that leaves fewer than four ticks
  let k = steps.findIndex((s) => s >= raw);
  if (ticksFor(steps[k]).length < 4) k--;
  const step = steps[k];
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return ticksFor(step).map((v) => ({ v, label: v.toFixed(decimals).replace("-", "−") }));
}
function logTicks(lo, hi) {
  const ticks = [];
  for (let k = Math.ceil(Math.log10(lo) - 1e-9); k <= Math.floor(Math.log10(hi) + 1e-9); k++) {
    const label = k >= -3 && k <= 4 ? String(Number(Math.pow(10, k).toPrecision(1))) : `10^{${k}}`;
    ticks.push({ v: Math.pow(10, k), label });
  }
  return ticks;
}

// marks are stored as functions and drawn at render time, once the layout (legend rows, panel stacking) is known
function Axes(fig, o) {
  this.fig = fig;
  this.o = o;
  this.xlim = o.xlim;
  this.ylim = o.ylim;
  this.xlog = !!o.xlog;
  this.ylog = !!o.ylog;
  this.marks = [];
  this.items = []; // legend entries
  this.legendRows = [];
  this.legendH = 0;
  this.id = `clip${fig.axesList.length}`;
  fig.axesList.push(this);
}
const A = Axes.prototype;
const LEGEND = { size: 12.5, row: 18, swatch: 24, gap: 16 };
// horizontal extent and legend rows for a panel occupying the given box
A._measure = function (box) {
  const o = this.o;
  const narrow = this.fig.narrow;
  const m = Object.assign(
    { left: narrow ? 58 : 64, right: narrow ? 12 : 16, top: o.xTop ? 56 + (o.title ? 22 : 0) : o.title ? 34 : 14, bottom: o.xTop ? 14 : o.xlabel ? 50 : 30 },
    o.margin || {}
  );
  this.m = m;
  this.box = box;
  this.x0 = box[0] + m.left;
  this.w = box[2] - m.left - m.right;
  this.legendRows = [];
  if (this.legendOn && this.items.length) {
    const avail = this.w + m.right - 4;
    let row = [];
    let x = 0;
    for (const it of this.items) {
      const width = LEGEND.swatch + 6 + textWidth(it.label, LEGEND.size);
      if (row.length && x + width > avail) {
        this.legendRows.push(row);
        row = [];
        x = 0;
      }
      row.push({ it, x });
      x += width + LEGEND.gap;
    }
    this.legendRows.push(row);
  }
  this.legendH = this.legendRows.length ? this.legendRows.length * LEGEND.row + 6 : 0;
};
// vertical position; extra is the space reserved for the legends of this row of panels
A._place = function (y, extra) {
  this.by = y;
  this.y0 = y + this.m.top + extra;
  this.h = this.box[3] - this.m.top - this.m.bottom;
};
A.px = function (x) {
  const [a, b] = this.xlim;
  const t = this.xlog ? (Math.log(x) - Math.log(a)) / (Math.log(b) - Math.log(a)) : (x - a) / (b - a);
  return this.x0 + t * this.w;
};
A.py = function (y) {
  const [a, b] = this.ylim;
  const t = this.ylog ? (Math.log(Math.max(y, 1e-300)) - Math.log(a)) / (Math.log(b) - Math.log(a)) : (y - a) / (b - a);
  return this.y0 + this.h - Math.max(-3, Math.min(4, t)) * this.h;
};
A.xInv = function (px) {
  const t = (px - this.x0) / this.w;
  const [a, b] = this.xlim;
  return this.xlog ? Math.exp(Math.log(a) + t * (Math.log(b) - Math.log(a))) : a + t * (b - a);
};
A.yInv = function (py) {
  const t = (this.y0 + this.h - py) / this.h;
  const [a, b] = this.ylim;
  return this.ylog ? Math.exp(Math.log(a) + t * (Math.log(b) - Math.log(a))) : a + t * (b - a);
};
A.contains = function (px, py) {
  return px >= this.x0 && px <= this.x0 + this.w && py >= this.y0 && py <= this.y0 + this.h;
};
A._add = function (svg, legend) {
  if (this._drawing) this._svg = svg;
  else if (legend && legend.label) this.items.push(legend);
  return this;
};
const dashOf = (d) => (d ? ` stroke-dasharray="${d === true ? "6 4" : d}"` : "");

A.line = function (xs, ys, s = {}) {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    const ok = Number.isFinite(xs[i]) && Number.isFinite(ys[i]) && !(this.ylog && ys[i] <= 0);
    if (!ok) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${r2(this.px(xs[i]))} ${r2(this.py(ys[i]))}`;
    pen = true;
  }
  const width = s.width || 2;
  return this._add(
    `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="${s.dash ? "butt" : "round"}"${dashOf(s.dash)}${s.opacity ? ` opacity="${s.opacity}"` : ""}/>`,
    { label: s.label, kind: "line", color: s.color, width, dash: s.dash }
  );
};
A.segments = function (list, s = {}) {
  const d = list.map((q) => `M${r2(this.px(q[0]))} ${r2(this.py(q[1]))}L${r2(this.px(q[2]))} ${r2(this.py(q[3]))}`).join("");
  return this._add(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1}" stroke-linecap="round"/>`);
};
// histogram outline or filled histogram; baseline may be a number or an array
A.steps = function (edges, values, s = {}) {
  const base = (i) => (Array.isArray(s.baseline) ? s.baseline[i] : s.baseline !== undefined ? s.baseline : this.ylog ? this.ylim[0] : 0);
  let top = "";
  for (let i = 0; i < values.length; i++) {
    const y = r2(this.py(values[i]));
    top += `${i ? "L" : "M"}${r2(this.px(edges[i]))} ${y}L${r2(this.px(edges[i + 1]))} ${y}`;
  }
  let svg = "";
  if (s.fill) {
    let d = top;
    for (let i = values.length - 1; i >= 0; i--) {
      const y = r2(this.py(base(i)));
      d += `L${r2(this.px(edges[i + 1]))} ${y}L${r2(this.px(edges[i]))} ${y}`;
    }
    svg += `<path d="${d}Z" fill="${s.hatch ? `url(#${this.fig.hatch(s.color)})` : s.color}" fill-opacity="${s.fillOpacity || 1}" stroke="none"/>`;
  }
  if (!s.fill || s.outline) {
    svg += `<path d="${top}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="miter"${dashOf(s.dash)}/>`;
  }
  return this._add(svg, {
    label: s.label, kind: s.fill ? "box" : "line", color: s.color, width: s.width || 2, dash: s.dash,
    opacity: s.fillOpacity, hatch: s.hatch,
  });
};
A.band = function (xs, lo, hi, s = {}) {
  let d = "";
  for (let i = 0; i < xs.length; i++) d += `${i ? "L" : "M"}${r2(this.px(xs[i]))} ${r2(this.py(hi[i]))}`;
  for (let i = xs.length - 1; i >= 0; i--) d += `L${r2(this.px(xs[i]))} ${r2(this.py(lo[i]))}`;
  return this._add(`<path d="${d}Z" fill="${s.color}" fill-opacity="${s.opacity || 1}" stroke="none"/>`, {
    label: s.label, kind: "box", color: s.color, opacity: s.opacity,
  });
};
A.rect = function (xa, ya, xb, yb, s = {}) {
  const x = Math.min(this.px(xa), this.px(xb));
  const y = Math.min(this.py(ya), this.py(yb));
  const w = Math.abs(this.px(xb) - this.px(xa));
  const h = Math.abs(this.py(yb) - this.py(ya));
  return this._add(
    `<rect x="${r2(x)}" y="${r2(y)}" width="${r2(w)}" height="${r2(h)}" fill="${s.fill || "none"}" fill-opacity="${s.opacity || 1}" stroke="${s.stroke || "none"}" stroke-width="${s.strokeWidth || 1.5}"/>`,
    { label: s.label, kind: "box", color: s.fill || "none", stroke: s.stroke, opacity: s.opacity }
  );
};
function markerSvg(kind, x, y, size, fill, stroke, strokeWidth) {
  const st = `fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"`;
  if (kind === "s") return `<rect x="${r2(x - size)}" y="${r2(y - size)}" width="${2 * size}" height="${2 * size}" ${st}/>`;
  if (kind === "x") {
    const a = size;
    return `<path d="M${r2(x - a)} ${r2(y - a)}L${r2(x + a)} ${r2(y + a)}M${r2(x - a)} ${r2(y + a)}L${r2(x + a)} ${r2(y - a)}" fill="none" stroke="${stroke}" stroke-width="${Math.max(strokeWidth, 2.5)}" stroke-linecap="round"/>`;
  }
  if (kind === "star") {
    let d = "";
    for (let i = 0; i < 10; i++) {
      const rad = i % 2 ? size * 0.45 : size * 1.25;
      const ang = (Math.PI * i) / 5 - Math.PI / 2;
      d += `${i ? "L" : "M"}${r2(x + rad * Math.cos(ang))} ${r2(y + rad * Math.sin(ang))}`;
    }
    return `<path d="${d}Z" ${st} stroke-linejoin="round"/>`;
  }
  return `<circle cx="${r2(x)}" cy="${r2(y)}" r="${size}" ${st}/>`;
}
// error bars: number, array (symmetric) or [low array, high array]
const errAt = (e, i) => (e === undefined ? null : Array.isArray(e) ? (Array.isArray(e[0]) ? [e[0][i], e[1][i]] : [e[i], e[i]]) : [e, e]);
A.points = function (xs, ys, s = {}) {
  const size = s.size || 5;
  const stroke = s.stroke || s.color;
  const fill = s.fill || s.color;
  const sw = s.strokeWidth || 1.5;
  let svg = "";
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue;
    const x = this.px(xs[i]);
    const y = this.py(ys[i]);
    const ey = errAt(s.yerr, i);
    const ex = errAt(s.xerr, i);
    if (ey) svg += `<line x1="${r2(x)}" x2="${r2(x)}" y1="${r2(this.py(ys[i] - ey[0]))}" y2="${r2(this.py(ys[i] + ey[1]))}" stroke="${s.errColor || s.color}" stroke-width="${s.errWidth || 2}"/>`;
    if (ex) svg += `<line y1="${r2(y)}" y2="${r2(y)}" x1="${r2(this.px(xs[i] - ex[0]))}" x2="${r2(this.px(xs[i] + ex[1]))}" stroke="${s.errColor || s.color}" stroke-width="${s.errWidth || 2}"/>`;
    svg += markerSvg(s.marker || "o", x, y, size, fill, stroke, sw);
  }
  return this._add(svg, { label: s.label, kind: "marker", marker: s.marker || "o", color: fill, stroke, strokeWidth: sw });
};
A.hline = function (y, s = {}) {
  return this._add(`<line x1="${this.x0}" x2="${this.x0 + this.w}" y1="${r2(this.py(y))}" y2="${r2(this.py(y))}" stroke="${s.color || "var(--axis)"}" stroke-width="${s.width || 1}"${dashOf(s.dash)}/>`,
    { label: s.label, kind: "line", color: s.color, width: s.width || 1, dash: s.dash });
};
A.vline = function (x, s = {}) {
  return this._add(`<line y1="${this.y0}" y2="${this.y0 + this.h}" x1="${r2(this.px(x))}" x2="${r2(this.px(x))}" stroke="${s.color || "var(--axis)"}" stroke-width="${s.width || 1}"${dashOf(s.dash)}/>`,
    { label: s.label, kind: "line", color: s.color, width: s.width || 1, dash: s.dash });
};
A.hspan = function (ya, yb, s = {}) {
  const top = Math.min(this.py(ya), this.py(yb));
  return this._add(`<rect x="${this.x0}" width="${this.w}" y="${r2(top)}" height="${r2(Math.abs(this.py(ya) - this.py(yb)))}" fill="${s.color}" fill-opacity="${s.opacity || 1}"/>`,
    { label: s.label, kind: "box", color: s.color, opacity: s.opacity });
};
A.vspan = function (xa, xb, s = {}) {
  const left = Math.min(this.px(xa), this.px(xb));
  return this._add(`<rect y="${this.y0}" height="${this.h}" x="${r2(left)}" width="${r2(Math.abs(this.px(xa) - this.px(xb)))}" fill="${s.color}" fill-opacity="${s.opacity || 1}"/>`,
    { label: s.label, kind: "box", color: s.color, opacity: s.opacity });
};
A.text = function (x, y, str, s = {}) {
  const size = s.size || 13;
  return this._add(`<text x="${r2(this.px(x) + (s.dx || 0))}" y="${r2(this.py(y) + (s.dy || 0))}" text-anchor="${s.anchor || "start"}" dominant-baseline="${s.baseline || "middle"}" font-size="${size}" fill="${s.color || "var(--ink)"}"${s.bold ? ' font-weight="600"' : ""}>${rich(str, size)}</text>`);
};
A.image = function (href, xa, xb, ya, yb) {
  return this._add(`<image href="${href}" x="${r2(this.px(xa))}" y="${r2(this.py(yb))}" width="${r2(this.px(xb) - this.px(xa))}" height="${r2(this.py(ya) - this.py(yb))}" preserveAspectRatio="none"/>`);
};
A.legend = function () {
  this.legendOn = true;
  return this;
};

// marks are declared before the layout is known: declaring one only registers its legend entry,
// the SVG is produced by calling the method again at render time
for (const name of ["line", "segments", "steps", "band", "rect", "points", "hline", "vline", "hspan", "vspan", "text", "image"]) {
  const make = A[name];
  A[name] = function (...args) {
    make.apply(this, args);
    this.marks.push(() => {
      make.apply(this, args);
      return this._svg;
    });
    return this;
  };
}

// legend in rows between the title and the plot area
A._legendSvg = function () {
  const top = this.by + (this.o.title ? 30 : 6);
  let svg = "";
  this.legendRows.forEach((row, k) => {
    const cy = top + LEGEND.row * (k + 0.5);
    for (const { it, x } of row) {
      const cx = this.x0 + x;
      const sw = LEGEND.swatch;
      if (it.kind === "line") {
        svg += `<line x1="${r2(cx)}" x2="${r2(cx + sw)}" y1="${r2(cy)}" y2="${r2(cy)}" stroke="${it.color || "var(--axis)"}" stroke-width="${Math.min(it.width, 3)}"${dashOf(it.dash)}/>`;
      } else if (it.kind === "box") {
        const fill = it.hatch ? `url(#${this.fig.hatch(it.color)})` : it.color;
        svg += `<rect x="${r2(cx)}" y="${r2(cy - 6)}" width="${sw}" height="12" fill="${fill}" fill-opacity="${it.opacity || 1}" stroke="${it.stroke || "none"}" stroke-width="1.5"/>`;
      } else {
        svg += markerSvg(it.marker, cx + sw / 2, cy, 4.5, it.color, it.stroke, it.strokeWidth);
      }
      svg += `<text x="${r2(cx + sw + 6)}" y="${r2(cy)}" dominant-baseline="central" font-size="${LEGEND.size}" fill="var(--ink-2)">${rich(it.label, LEGEND.size)}</text>`;
    }
  });
  return svg;
};

A._render = function () {
  const o = this.o;
  const { x0, y0, w, h } = this;
  const size = 12.5;
  const [bx, , bw] = this.box || [x0, 0, w];
  // text centred on the plot area, moved inwards (and shrunk) if it would leave the panel
  const centred = (str, y, fontSize, bold, min) => {
    const fs = fitted(str, fontSize, bw - 8, bold, min);
    const half = textWidth(str, fs, bold) / 2;
    const cx = Math.min(Math.max(x0 + w / 2, bx + half + 4), bx + bw - half - 4);
    return `<text x="${r2(cx)}" y="${r2(y)}" text-anchor="middle" font-size="${r2(fs)}"${bold ? ' font-weight="600"' : ""} fill="var(--ink)">${rich(str, fs)}</text>`;
  };
  let svg = "";
  const xt = o.xticks ? o.xticks.map((t) => (typeof t === "number" ? { v: t, label: String(t).replace("-", "−") } : t)) : this.xlog ? logTicks(...this.xlim) : niceTicks(this.xlim[0], this.xlim[1], Math.max(3, Math.round(w / 95)));
  const yt = o.yticks ? o.yticks.map((t) => (typeof t === "number" ? { v: t, label: String(t).replace("-", "−") } : t)) : this.ylog ? logTicks(...this.ylim) : niceTicks(this.ylim[0], this.ylim[1], Math.max(3, Math.round(h / 60)));
  if (o.grid !== false && !o.overlay) {
    for (const t of xt) svg += `<line x1="${r2(this.px(t.v))}" x2="${r2(this.px(t.v))}" y1="${y0}" y2="${y0 + h}" stroke="var(--grid)" stroke-width="1"/>`;
    for (const t of yt) svg += `<line y1="${r2(this.py(t.v))}" y2="${r2(this.py(t.v))}" x1="${x0}" x2="${x0 + w}" stroke="var(--grid)" stroke-width="1"/>`;
  }
  this._drawing = true;
  const marks = this.marks.map((draw) => draw()).join("");
  this._drawing = false;
  svg += `<clipPath id="${this.fig.uid}${this.id}"><rect x="${x0}" y="${y0}" width="${w}" height="${h}"/></clipPath>`;
  svg += `<g clip-path="url(#${this.fig.uid}${this.id})">${marks}</g>`;
  // open axes: left and bottom spine only, a full frame where a second axis sits on top
  if (!o.overlay) {
    svg += o.xTop
      ? `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="none" stroke="var(--axis)" stroke-width="1"/>`
      : `<path d="M${x0} ${y0}V${y0 + h}H${x0 + w}" fill="none" stroke="var(--axis)" stroke-width="1"/>`;
  }
  // x ticks: below the plot, or above for an overlaid top axis
  const top = !!o.xTop;
  for (const t of xt) {
    const x = this.px(t.v);
    if (x < x0 - 0.5 || x > x0 + w + 0.5) continue;
    const ya = top ? y0 : y0 + h;
    svg += `<line x1="${r2(x)}" x2="${r2(x)}" y1="${ya}" y2="${ya + (top ? -5 : 5)}" stroke="var(--axis)"/>`;
    svg += `<text x="${r2(x)}" y="${ya + (top ? -9 : 19)}" text-anchor="middle" font-size="${size}" fill="var(--ink-2)">${rich(t.label, size)}</text>`;
  }
  if (o.xlabel) svg += centred(o.xlabel, top ? y0 - 28 : y0 + h + 41, 13.5, false, 10.5);
  if (!o.overlay) {
    for (const t of yt) {
      const y = this.py(t.v);
      if (y < y0 - 0.5 || y > y0 + h + 0.5) continue;
      svg += `<line x1="${x0 - 5}" x2="${x0}" y1="${r2(y)}" y2="${r2(y)}" stroke="var(--axis)"/>`;
      svg += `<text x="${x0 - 9}" y="${r2(y)}" text-anchor="end" dominant-baseline="central" font-size="${size}" fill="var(--ink-2)">${rich(t.label, size)}</text>`;
    }
    if (o.ylabel) {
      const lx = x0 - (o.ylabelOffset || Math.min(46, this.m.left - 16));
      const fs = fitted(o.ylabel, 13.5, h, false, 10.5);
      svg += `<text transform="translate(${lx} ${y0 + h / 2}) rotate(-90)" text-anchor="middle" font-size="${r2(fs)}" fill="var(--ink)">${rich(o.ylabel, fs)}</text>`;
    }
    if (o.title) svg += centred(o.title, this.by + 21, 14.5, true, 11);
    svg += this._legendSvg();
  }
  return svg;
};

// width available for the figure inside its container
function available(container) {
  const cs = getComputedStyle(container);
  return container.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
}
const NARROW = 700;
// changes whenever a figure has to be drawn again after its container was resized
const layoutKey = (container) => {
  const width = available(container);
  return width > 0 && width < NARROW ? Math.round(width) : 0;
};

let figureCount = 0;
function Figure(container, o) {
  this.container = container;
  // in a narrow container the figure is drawn at the container width (so text keeps its size)
  // and the panels are stacked instead of placed in the boxes given by the page
  const key = layoutKey(container);
  this.narrow = key > 0;
  this.W = this.narrow ? Math.max(300, key) : o.width;
  this.H = o.height;
  this.axesList = [];
  this.defs = {};
  this.uid = `f${figureCount++}`;
}
Figure.prototype.axes = function (o) {
  return new Axes(this, o);
};
// second set of marks with its own x range, drawn over an existing axes (same box, same y range)
Figure.prototype.overlay = function (ax, o) {
  const over = new Axes(this, Object.assign({}, o, { overlay: true, box: ax.o.box, ylim: ax.ylim }));
  over.base = ax;
  return over;
};
Figure.prototype._layout = function () {
  const panels = this.axesList.filter((a) => !a.o.overlay);
  let rows;
  if (this.narrow) rows = panels.map((a) => [a]);
  else {
    const byTop = new Map();
    for (const a of panels) byTop.set(a.o.box[1], (byTop.get(a.o.box[1]) || []).concat([a]));
    rows = [...byTop.keys()].sort((p, q) => p - q).map((k) => byTop.get(k));
  }
  let y = 0;
  for (const row of rows) {
    for (const a of row) {
      const [bx, , bw, bh] = a.o.box;
      a._measure(this.narrow ? [0, 0, this.W, Math.min(bh, Math.max(260, Math.round(this.W * 1.05)))] : [bx, 0, bw, bh]);
    }
    const extra = Math.max(...row.map((a) => a.legendH));
    for (const a of row) a._place(y, extra);
    y += Math.max(...row.map((a) => a.box[3])) + extra;
  }
  this.H = y;
  for (const a of this.axesList) {
    if (!a.o.overlay) continue;
    for (const k of ["m", "box", "by", "x0", "y0", "w", "h"]) a[k] = a.base[k];
  }
};
Figure.prototype.hatch = function (color) {
  const id = `${this.uid}h${color.replace(/[^a-z0-9]/gi, "")}`;
  this.defs[id] = `<pattern id="${id}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="${color}" stroke-width="1.5"/></pattern>`;
  return id;
};
Figure.prototype.render = function () {
  this._layout();
  const body = this.axesList.map((a) => a._render()).join("");
  this.container.innerHTML = `<svg viewBox="0 0 ${this.W} ${this.H}" role="img" xmlns="http://www.w3.org/2000/svg"><defs>${Object.values(this.defs).join("")}</defs>${body}</svg>`;
  this.container._figure = this;
  attachReadout(this.container);
  return this;
};
// position of a pointer event in figure coordinates
Figure.prototype.locate = function (event) {
  const svg = this.container.querySelector("svg");
  const box = svg.getBoundingClientRect();
  return [((event.clientX - box.left) / box.width) * this.W, ((event.clientY - box.top) / box.height) * this.H];
};
Figure.prototype.axesAt = function (px, py) {
  return this.axesList.find((a) => !a.o.overlay && a.contains(px, py));
};

// readout of the data coordinates under the pointer
function attachReadout(container) {
  if (container._readout) return;
  const tip = document.createElement("div");
  tip.className = "readout";
  tip.hidden = true;
  container._readout = tip;
  container.style.position = "relative";
  container.addEventListener("pointermove", (event) => {
    const fig = container._figure;
    const [px, py] = fig.locate(event);
    const ax = fig.axesAt(px, py);
    if (!ax || ax.o.readout === false) {
      tip.hidden = true;
      return;
    }
    const fmt = (v, lim) => {
      const span = Math.abs(lim[1] - lim[0]);
      return v.toFixed(Math.max(0, Math.min(4, 2 - Math.floor(Math.log10(span))))).replace("-", "−");
    };
    if (ax.o.tip) {
      // the panel describes what is under the pointer itself (may be several lines)
      const text = ax.o.tip(ax.xInv(px), ax.yInv(py));
      if (!text) {
        tip.hidden = true;
        return;
      }
      tip.textContent = text;
    } else {
      const fx = ax.o.readoutX ? ax.o.readoutX(ax.xInv(px)) : fmt(ax.xInv(px), ax.xlim);
      const fy = ax.ylog ? ax.yInv(py).toPrecision(2) : fmt(ax.yInv(py), ax.ylim);
      tip.textContent = `${fx}, ${fy}`;
    }
    if (!tip.parentNode || tip.parentNode !== container) container.appendChild(tip);
    tip.hidden = false;
    const box = container.getBoundingClientRect();
    const x = event.clientX - box.left;
    const flip = x + 14 + tip.offsetWidth > box.width;
    tip.style.left = `${flip ? Math.max(0, x - 14 - tip.offsetWidth) : x + 14}px`;
    tip.style.top = `${event.clientY - box.top - 16 - tip.offsetHeight}px`;
  });
  container.addEventListener("pointerleave", () => (tip.hidden = true));
}

// filled regions of a 2D field as an image: z[i][j] = z(x[i], y[j]), colors given as [r, g, b] per band
function bandImage(z, levels, colors) {
  const nx = z.length;
  const ny = z[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = z[i][ny - 1 - j];
      const k = levels.findIndex((l) => v < l);
      const p = 4 * (j * nx + i);
      if (k >= 0) {
        img.data[p] = colors[k][0];
        img.data[p + 1] = colors[k][1];
        img.data[p + 2] = colors[k][2];
        img.data[p + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

const figure = (container, o) => new Figure(container, o);

export { figure, bandImage, layoutKey, rich };
