// Profile likelihood scan of one parameter.
import { figure } from "./plot.js";

// scan: {values, twiceDeltaNll}; best, unc: fit result for the parameter (unc > 0 adds the parabola)
function drawScan(container, scan, { name, best, unc }) {
  const x = scan.values;
  const y = scan.twiceDeltaNll;
  const top = Math.min(Math.max(...y.filter(Number.isFinite), 4.5), 25);
  const fig = figure(container, { width: 760, height: 430 });
  const a = fig.axes({ box: [0, 0, 760, 430], xlim: [x[0], x[x.length - 1]], ylim: [0, top * 1.05], xlabel: name, ylabel: "−2 Δln L" });
  a.hline(1, { color: "var(--grey)", dash: "2 4" });
  a.hline(4, { color: "var(--grey)", dash: "2 4" });
  if (unc > 0) {
    const xs = Array.from({ length: 121 }, (_, i) => x[0] + ((x[x.length - 1] - x[0]) * i) / 120);
    a.line(xs, xs.map((v) => ((v - best) / unc) ** 2), { color: "var(--s2)", dash: true, label: "parabola from the covariance matrix" });
  }
  a.line(x, y, { color: "var(--s1)", width: 2.5, label: "profile likelihood" });
  a.points(x, y, { color: "var(--s1)", size: 3, stroke: "var(--plot-bg)", strokeWidth: 1 });
  a.legend();
  return fig.render();
}

export { drawScan };
