// Pulls and Correlations tabs.
import { drawCorrelations, drawPulls, esc, num, pull } from "@histopresto/viz";
import { App } from "../app.js";
const isGamma = (p) => p.types.includes("staterror") || p.types.includes("shapesys");

App.view({
  id: "pulls",
  title: "Pulls",
  needsFit: true,
  render(container, state) {
    const s = state.summary;
    const f = state.fit;
    const constrained = s.params.map((p, i) => ({ p, i })).filter(({ p }) => p.constraint !== "none" && !p.fixed);
    const nGamma = constrained.filter(({ p }) => isGamma(p)).length;
    const gammas = App.ui("pulls.gammas", constrained.length <= 60);
    const filter = App.ui("pulls.filter", "").toLowerCase();
    const sort = App.ui("pulls.sort", "model");
    let rows = constrained
      .filter(({ p }) => (gammas || !isGamma(p)) && (!filter || p.name.toLowerCase().includes(filter)))
      .map(({ p, i }) => ({ p, i, name: p.name, value: f.pars[i], unc: f.unc[i], ...pull(p, f.pars[i], f.unc[i]) }));
    if (sort === "pull") rows.sort((a, b) => Math.abs(b.pull) - Math.abs(a.pull));
    if (sort === "constraint") rows.sort((a, b) => a.err - b.err);

    const free = s.params.map((p, i) => ({ p, i })).filter(({ p }) => p.constraint === "none" && !p.fixed);
    container.innerHTML =
      App.toolbar(
        `<label>filter <input id="pulls-filter" type="search" data-ui="pulls.filter" value="${esc(App.ui("pulls.filter", ""))}" placeholder="part of a name"></label>` +
          `<label>order <select data-ui="pulls.sort">${[["model", "as in the model"], ["pull", "largest pull first"], ["constraint", "most constrained first"]].map(([v, l]) => `<option value="${v}"${v === sort ? " selected" : ""}>${l}</option>`).join("")}</select></label>` +
          (nGamma ? `<label><input type="checkbox" data-ui="pulls.gammas"${gammas ? " checked" : ""}> include ${nGamma} γ parameters (staterror, shapesys)</label>` : "") +
          `<span>${rows.length} shown</span>`
      ) +
      (rows.length ? `<div class="figure" id="pull-figure" style="max-width: 860px"></div>` : `<p class="empty">No constrained parameters match.</p>`) +
      `<p class="note">Each point is (best fit − nominal value) / pre-fit uncertainty, the bar is the post-fit uncertainty in the same units. A bar shorter than the green band means the data constrain the parameter beyond its pre-fit uncertainty.</p>` +
      (free.length
        ? `<h2>Parameters without a constraint term</h2><div class="table-wrap"><table class="data"><tr><th>parameter</th><th class="num">best fit</th><th class="num">uncertainty</th><th class="num">range</th><th></th></tr>` +
          free
            .map(({ p, i }) => `<tr><td class="mono">${esc(p.name)}</td><td class="num">${num(f.pars[i])}</td><td class="num">${f.atBound[i] ? "–" : `± ${num(f.unc[i], 2)}`}</td><td class="num">${num(p.lower)} … ${num(p.upper)}</td><td class="dim">${f.atBound[i] ? "at the edge of its range" : ""}</td></tr>`)
            .join("") +
          `</table></div>`
        : "");
    if (rows.length) drawPulls(container.querySelector("#pull-figure"), rows);
  },
});

App.view({
  id: "correlations",
  title: "Correlations",
  needsFit: true,
  render(container, state) {
    const s = state.summary;
    const f = state.fit;
    if (!f.correlation) {
      container.innerHTML = `<p class="empty">The fit did not produce a covariance matrix.</p>`;
      return;
    }
    const k = f.inner.length;
    const MAX = 40;
    const threshold = App.ui("corr.threshold", 0.3);
    const strongest = new Float64Array(k);
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) if (a !== b) strongest[a] = Math.max(strongest[a], Math.abs(f.correlation[a * k + b]));
    let keep = [];
    for (let a = 0; a < k; a++) if (strongest[a] >= threshold || f.inner[a] === s.poiIndex) keep.push(a);
    const passing = keep.length;
    if (keep.length > MAX) {
      keep.sort((a, b) => (f.inner[b] === s.poiIndex) - (f.inner[a] === s.poiIndex) || strongest[b] - strongest[a]);
      keep = keep.slice(0, MAX).sort((a, b) => a - b);
    }
    const n = keep.length;
    container.innerHTML =
      App.toolbar(
        `<label>show parameters with a correlation of at least <input id="corr-threshold" type="number" min="0" max="1" step="0.05" data-ui="corr.threshold" value="${threshold}"></label>` +
          `<span>${passing > MAX ? `${MAX} strongest of ${passing}` : `${n}`} of ${k} parameters</span>`
      ) +
      (n > 1
        ? `<div class="figure" id="corr-figure"></div>` +
          `<div class="scale"><span>−1</span><span class="bar"></span><span>+1</span></div>`
        : `<p class="empty">No pair of parameters is correlated this strongly. Lower the threshold to see more.</p>`);
    if (n < 2) return;

    const names = keep.map((a) => s.params[f.inner[a]].name);
    drawCorrelations(container.querySelector("#corr-figure"), names, (i, j) => f.correlation[keep[i] * k + keep[j]]);
  },
});
