// Data/MC figure of one channel: stacked expectation with its uncertainty, data, and the ratio below.
import { figure } from "./plot.js";
import { yieldFmt } from "./format.js";

// channel: {name, data, total, totalUnc} (one entry per bin)
// layers: [{name, colour, yields}] in stacking order, bottom first
// options: mode ("pre" | "post", used in the readout), logy, wide (full width instead of half)
function drawYields(container, ch, layers, { mode = "post", logy = false, wide = false } = {}) {
  const n = ch.data.length;
  const W = wide ? 860 : 430;
  const fig = figure(container, { width: W, height: 420 });
  const edges = Array.from({ length: n + 1 }, (_, i) => i);
  const centres = Array.from({ length: n }, (_, i) => i + 0.5);
  const err = ch.data.map((v) => Math.sqrt(v));
  // no band where the uncertainty is unknown (a fit without covariance matrix)
  const unc = ch.totalUnc.map((u) => (Number.isFinite(u) ? u : 0));
  const top = Math.max(...ch.total.map((v, b) => v + unc[b]), ...ch.data.map((v, b) => v + err[b]));
  const positive = ch.total.filter((v) => v > 0);
  const low = logy ? Math.pow(10, Math.floor(Math.log10(Math.max(Math.min(...positive, ...ch.data.filter((v) => v > 0)) / 3, 1e-3)))) : 0;
  const stride = Math.ceil(n / (wide ? 20 : 10));
  const xticks = centres.filter((_, i) => i % stride === 0).map((v) => ({ v, label: String(Math.floor(v) + 1) }));
  const tip = (x) => {
    const b = Math.floor(x);
    if (b < 0 || b >= n) return null;
    const lines = [`bin ${b + 1}`, `data  ${yieldFmt(ch.data[b])}`, `${mode}-fit  ${yieldFmt(ch.total[b])} ± ${yieldFmt(ch.totalUnc[b])}`];
    for (const l of layers.slice().reverse()) lines.push(`  ${l.name}  ${yieldFmt(l.yields[b])}`);
    return lines.join("\n");
  };

  const a = fig.axes({
    box: [0, 0, W, 290], xlim: [0, n], ylim: [low, logy ? Math.pow(10, Math.ceil(Math.log10(top * 1.5))) : top * 1.12 || 1], ylog: logy,
    title: ch.name, ylabel: "events", xticks: [], grid: false, margin: { bottom: 8 }, tip,
  });
  // stacked from the bottom; drawn top layer first so that the legend reads like the stack
  const cumulative = [];
  let running = new Array(n).fill(0);
  for (const l of layers) {
    const base = running.map((v) => (logy ? Math.max(v, low) : v));
    running = running.map((v, b) => v + l.yields[b]);
    cumulative.push({ l, base, top: running.slice() });
  }
  for (const { l, base, top: t } of cumulative.slice().reverse()) a.steps(edges, t, { fill: true, color: l.colour, baseline: base, label: l.name });
  a.steps(edges, ch.total.map((v, b) => v + unc[b]), { fill: true, hatch: true, color: "var(--ink-2)", baseline: ch.total.map((v, b) => Math.max(v - unc[b], low)), label: "uncertainty" });
  a.points(centres, ch.data.map((v) => (logy && v <= 0 ? NaN : v)), { color: "var(--ink)", size: 3.5, yerr: [ch.data.map((v, b) => (logy ? Math.min(err[b], v - low) : err[b])), err], errWidth: 1.5, stroke: "var(--plot-bg)", strokeWidth: 1, label: "data" });
  a.legend();

  const ratio = ch.data.map((v, b) => (ch.total[b] > 0 ? v / ch.total[b] : NaN));
  const ratioErr = err.map((e, b) => (ch.total[b] > 0 ? e / ch.total[b] : 0));
  const band = unc.map((u, b) => (ch.total[b] > 0 ? u / ch.total[b] : 0));
  const worst = Math.max(...ratio.map((r, b) => (Number.isFinite(r) ? Math.abs(r - 1) + ratioErr[b] : 0)), ...band);
  const span = Math.min(Math.max(1.15 * worst, 0.1), 1);
  const r = fig.axes({
    box: [0, 290, W, 130], xlim: [0, n], ylim: [1 - span, 1 + span], xlabel: "bin", ylabel: "data / model", xticks, margin: { top: 6 }, tip,
  });
  r.steps(edges, band.map((u) => 1 + u), { fill: true, hatch: true, color: "var(--ink-2)", baseline: band.map((u) => 1 - u) });
  r.hline(1, { color: "var(--axis)" });
  r.points(centres, ratio, { color: "var(--ink)", size: 3.5, yerr: ratioErr, errWidth: 1.5, stroke: "var(--plot-bg)", strokeWidth: 1 });
  return fig.render();
}

export { drawYields };
