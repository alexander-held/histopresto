// CLs as a function of the parameter of interest, observed and expected with bands.
import { figure } from "./plot.js";
import { num } from "./format.js";

// limit: {values, level, results: [{mu, cls, expected: [-2σ, -1σ, median, +1σ, +2σ]}]}
function drawLimit(container, limit, { poiName, logy = false }) {
  const x = limit.values;
  const exp = (k) => limit.results.map((r) => r.expected[k]);
  const fig = figure(container, { width: 760, height: 430 });
  const a = fig.axes({
    box: [0, 0, 760, 430], xlim: [0, x[x.length - 1]], ylim: logy ? [1e-3, 2] : [0, 1.05], ylog: logy, xlabel: poiName, ylabel: "CL_{s}",
    tip: (v) => {
      let i = 0;
      for (let j = 1; j < x.length; j++) if (Math.abs(x[j] - v) < Math.abs(x[i] - v)) i = j;
      const r = limit.results[i];
      return `${poiName} = ${num(r.mu)}\nobserved  ${num(r.cls, 2)}\nexpected  ${num(r.expected[2], 2)}`;
    },
  });
  a.band(x, exp(0), exp(4), { color: "var(--band-2)", label: "expected ±2σ" });
  a.band(x, exp(1), exp(3), { color: "var(--band-1)", label: "expected ±1σ" });
  a.line(x, exp(2), { color: "var(--ink)", dash: true, width: 1.5, label: "expected" });
  a.hline(limit.level, { color: "var(--s8)", width: 1.5, label: `CL_{s} = ${limit.level}` });
  a.line(x, limit.results.map((r) => r.cls), { color: "var(--ink)", width: 2, label: "observed" });
  a.points(x, limit.results.map((r) => r.cls), { color: "var(--ink)", size: 2.5 });
  a.legend();
  return fig.render();
}

export { drawLimit };
