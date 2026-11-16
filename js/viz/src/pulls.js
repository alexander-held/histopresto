// Pull plot: best-fit values of constrained parameters in units of their pre-fit uncertainty.
import { figure } from "./plot.js";
import { fmt, num, short } from "./format.js";

// (best fit - nominal) / pre-fit uncertainty for a constrained parameter
// param: {constraint: "Gaussian" | "Poisson" | "none", aux, prefitUnc}
function pull(param, value, unc) {
  const nominal = param.constraint === "Poisson" ? 1 : param.aux;
  return { pull: (value - nominal) / param.prefitUnc, err: unc / param.prefitUnc };
}

// rows: [{name, value, unc, pull, err}], drawn top to bottom
function drawPulls(container, rows) {
  const n = rows.length;
  const reach = Math.max(2.6, ...rows.map((r) => Math.abs(r.pull) + r.err + 0.2));
  const span = Math.ceil(reach);
  const fig = figure(container, { width: 860, height: 60 + 19 * n + 50 });
  const longest = Math.max(...rows.map((r) => short(r.name).length));
  const a = fig.axes({
    box: [0, 0, 860, 14 + 19 * n + 50], xlim: [-span, span], ylim: [-n + 0.5, 0.5], grid: false,
    xlabel: "(θ̂ − θ_{0}) / Δθ", margin: { left: Math.min(fig.narrow ? 170 : 330, 20 + 7.2 * longest), top: 14 },
    yticks: rows.map((r, k) => ({ v: -k, label: fig.narrow ? short(r.name, 22) : short(r.name) })),
    tip: (x, y) => {
      const r = rows[Math.round(-y)];
      return r ? `${r.name}\nbest fit  ${num(r.value)} ± ${num(r.unc, 2)}\npull  ${fmt(r.pull)},  constraint  ${fmt(r.err)}` : null;
    },
  });
  a.vspan(-2, 2, { color: "var(--band-2)" });
  a.vspan(-1, 1, { color: "var(--band-1)" });
  a.vline(0, { color: "var(--ink)", dash: "5 4" });
  a.points(rows.map((r) => r.pull), rows.map((_, k) => -k), { color: "var(--ink)", size: 3.5, xerr: rows.map((r) => r.err), errWidth: 2 });
  return fig.render();
}

export { pull, drawPulls };
