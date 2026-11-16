// Ranking plot: impact of nuisance parameters on the parameter of interest, with their pulls.
import { figure } from "./plot.js";
import { num, short } from "./format.js";

const UP = "var(--rank-up)";
const DOWN = "var(--rank-down)";

// rows: [{name, postfitUp, postfitDown, prefitUp, prefitDown, exact, pull: {pull, err} | null}], largest impact first.
// exact: false draws the impact lighter (an estimate rather than the result of refits).
function drawRanking(container, rows, poiName) {
  const n = rows.length;
  const span = 1.15 * Math.max(...rows.flatMap((r) => [r.postfitUp, r.postfitDown, r.prefitUp, r.prefitDown].map(Math.abs)), 1e-12);
  const fig = figure(container, { width: 900, height: 110 + 26 * n });
  const longest = Math.max(...rows.map((r) => short(r.name).length));
  const b = fig.axes({
    box: [0, 0, 900, 110 + 26 * n], xlim: [-span, span], ylim: [-n + 0.4, 0.6], xTop: true, xlabel: `impact on ${poiName}`,
    grid: false, margin: { left: Math.min(fig.narrow ? 170 : 330, 20 + 7.2 * longest), bottom: 50 },
    yticks: rows.map((r, k) => ({ v: -k, label: short(r.name, fig.narrow ? 22 : 44) })),
    tip: (x, y) => {
      const r = rows[Math.round(-y)];
      if (!r) return null;
      return `${r.name}\npost-fit  ${num(r.postfitUp, 2)} / ${num(r.postfitDown, 2)}\npre-fit  ${num(r.prefitUp, 2)} / ${num(r.prefitDown, 2)}\n${r.exact ? "from refits" : "estimated from the covariance"}`;
    },
  });
  rows.forEach((r, k) => {
    b.rect(0, -k - 0.34, r.postfitUp, -k + 0.34, { fill: UP, opacity: r.exact ? 0.95 : 0.55, label: k ? "" : "+1σ post-fit" });
    b.rect(0, -k - 0.34, r.postfitDown, -k + 0.34, { fill: DOWN, opacity: r.exact ? 0.95 : 0.55, label: k ? "" : "−1σ post-fit" });
    if (r.prefitUp || r.prefitDown) {
      b.rect(0, -k - 0.34, r.prefitUp, -k + 0.34, { stroke: UP, strokeWidth: 1.5, label: k ? "" : "+1σ pre-fit" });
      b.rect(0, -k - 0.34, r.prefitDown, -k + 0.34, { stroke: DOWN, strokeWidth: 1.5, label: k ? "" : "−1σ pre-fit" });
    }
  });
  b.legend();
  const pulls = fig.overlay(b, { xlim: [-2, 2], xlabel: "pull: (θ̂ − θ_{0}) / Δθ", xticks: [-2, -1, 0, 1, 2], yticks: [] });
  pulls.vline(-1, { color: "var(--grey)", dash: "2 4" });
  pulls.vline(1, { color: "var(--grey)", dash: "2 4" });
  pulls.vline(0, { color: "var(--ink)", dash: "5 4" });
  const constrained = rows.map((r, k) => ({ r, k })).filter(({ r }) => r.pull);
  pulls.points(constrained.map(({ r }) => r.pull.pull), constrained.map(({ k }) => -k), { color: "var(--ink)", size: 4, xerr: constrained.map(({ r }) => r.pull.err), errWidth: 2 });
  return fig.render();
}

export { drawRanking };
