// Profile likelihood scans of one and of two parameters.
import { figure } from "./plot.js";
import { contour, refine } from "./contour.js";
import { num, percent } from "./format.js";

const BANDS = ["var(--band-1)", "var(--band-2)"];

// scan: {values, twiceDeltaNll, reference, intervals: [{cl, level, lo, hi}]}, intervals ordered by cl;
// best, unc: fit result for the parameter (unc > 0 adds the parabola)
function drawScan(container, scan, { name, best, unc }) {
  const x = scan.values;
  const y = scan.twiceDeltaNll;
  const xlim = [x[0], x[x.length - 1]];
  const top = Math.min(Math.max(...y.filter(Number.isFinite), 4.5), 25);
  const ref = scan.reference || 0;
  const intervals = scan.intervals || [];
  const fig = figure(container, { width: 760, height: 430 });
  const a = fig.axes({ box: [0, 0, 760, 430], xlim, ylim: [0, top * 1.05], xlabel: name, ylabel: "−2 Δln L" });
  // each interval is shaded up to its level, the widest first
  const clip = (v) => Math.min(Math.max(v, xlim[0]), xlim[1]);
  const shades = intervals.map((iv, k) => [iv, BANDS[k] || BANDS[1]]).filter(([iv]) => !Number.isNaN(iv.lo)).reverse();
  for (const [iv, fill] of shades) a.rect(clip(iv.lo), ref, clip(iv.hi), ref + iv.level, { fill, label: `${percent(iv.cl)} CI` });
  for (const iv of intervals) a.hline(ref + iv.level, { color: "var(--grey)", dash: "2 4" });
  if (unc > 0) {
    const xs = Array.from({ length: 121 }, (_, i) => x[0] + ((x[x.length - 1] - x[0]) * i) / 120);
    a.line(xs, xs.map((v) => ((v - best) / unc) ** 2), { color: "var(--s2)", dash: true, label: "parabola from the covariance matrix" });
  }
  a.line(x, y, { color: "var(--s1)", width: 2.5, label: "profile likelihood" });
  a.points(x, y, { color: "var(--s1)", size: 3, stroke: "var(--plot-bg)", strokeWidth: 1 });
  a.legend();
  return fig.render();
}

// scan: {x, y, twiceDeltaNll[i][j], levels: [{cl, level}]}, levels ordered by cl; best: [x, y] of the fit
function drawScan2d(container, scan, { names, best }) {
  const fig = figure(container, { width: 640, height: 600 });
  const coarse = scan.twiceDeltaNll;
  const nearest = (v, grid) => grid.reduce((k, g, i) => (Math.abs(g - v) < Math.abs(grid[k] - v) ? i : k), 0);
  const a = fig.axes({
    box: [0, 0, 640, 600], xlim: [scan.x[0], scan.x[scan.x.length - 1]], ylim: [scan.y[0], scan.y[scan.y.length - 1]], xlabel: names[0], ylabel: names[1],
    margin: { left: 76 }, ylabelOffset: 58,
    tip: (u, v) => {
      const [i, j] = [nearest(u, scan.x), nearest(v, scan.y)];
      return `${names[0]} = ${num(scan.x[i])}\n${names[1]} = ${num(scan.y[j])}\n−2 Δln L = ${num(coarse[i][j])}`;
    },
  });
  const { x, y, z } = refine(scan.x, scan.y, coarse);
  // measured from the fit result, or from the lowest point of the scan should that lie below it
  const ref = Math.min(0, ...coarse.flat().filter(Number.isFinite));
  const levels = scan.levels.map((l, k) => ({ ...l, k })).reverse();
  for (const l of levels) a.paths(contour(x, y, z, ref + l.level, true), { fill: BANDS[l.k] || BANDS[1], label: `${percent(l.cl)} CL` });
  for (const l of levels) a.paths(contour(x, y, z, ref + l.level), { color: "var(--ink)", width: 1.5, dash: l.k ? "5 4" : null });
  a.points([best[0]], [best[1]], { color: "var(--ink)", marker: "x", size: 5, label: "best fit" });
  a.legend();
  return fig.render();
}

export { drawScan, drawScan2d };
