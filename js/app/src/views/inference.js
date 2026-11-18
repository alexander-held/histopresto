// Ranking, Scans and Limits tabs.
import { drawLimit, drawRanking, drawScan, drawScan2d, esc, fmt, num, percent, pull } from "@histopresto/viz";
import { App } from "../app.js";

App.view({
  id: "ranking",
  title: "Ranking",
  needsFit: true,
  needsPoi: true,
  render(container, state) {
    const s = state.summary;
    const f = state.fit;
    if (!f.impacts.length) {
      container.innerHTML = `<p class="empty">${f.atBound[s.poiIndex] ? `The best-fit value of ${esc(s.poiName)} sits at the edge of its range, so impacts on it cannot be evaluated. Widen the range to allow the fit to move.` : "There are no nuisance parameters to rank."}</p>`;
      return;
    }
    const entry = (r) => {
      const e = state.exact.get(r.index);
      return e || { index: r.index, postfitUp: r.postfit, postfitDown: -r.postfit, prefitUp: r.prefit, prefitDown: -r.prefit, exact: false };
    };
    const size = (e) => Math.max(Math.abs(e.postfitUp), Math.abs(e.postfitDown));
    // candidates by the estimate from the covariance matrix, then ordered by what is known best
    const rows = App.rankingCandidates().map(entry).sort((a, b) => size(b) - size(a));
    const missing = rows.filter((r) => !r.exact).map((r) => r.index);

    container.innerHTML =
      App.toolbar(
        `<label>show the <select data-ui="rank.n" data-num>${[10, 15, 20, 30, 50].map((v) => `<option${v === App.ui("rank.n", 15) ? " selected" : ""}>${v}</option>`).join("")}</select> leading parameters</label>` +
          `<button id="rank-run" data-job${missing.length ? "" : " disabled"}>${missing.length ? `Refit for exact impacts (${4 * missing.length} fits)` : "All impacts shown are from refits"}</button>`
      ) +
      `<div class="figure" id="rank-figure" style="max-width: 900px"></div>` +
      `<p class="note">Impact on ${esc(s.poiName)} when a nuisance parameter is moved by its pre-fit (outlined) or post-fit (filled) uncertainty. ` +
      (missing.length
        ? `${missing.length === rows.length ? "These are" : `${missing.length} of these are`} estimates from the covariance matrix of the fit (symmetric by construction); the refit holds each parameter at the shifted value and minimises again.`
        : `Each parameter was held at the shifted value and the fit repeated.`) +
      `</p>`;
    container.querySelector("#rank-run").onclick = () => App.actions.ranking(missing);

    const figureRows = rows.map((r) => {
      const p = s.params[r.index];
      return { ...r, name: p.name, pull: p.constraint === "none" ? null : pull(p, f.pars[r.index], f.unc[r.index]) };
    });
    drawRanking(container.querySelector("#rank-figure"), figureRows, s.poiName);
  },
});

const freeParams = (s) => s.params.map((p, i) => ({ p, i })).filter(({ p }) => !p.fixed);
const paramSelect = (s, key, index, skip) =>
  `<select data-ui="${key}" data-num style="max-width: 22rem">${freeParams(s).filter(({ i }) => i !== skip).map(({ p, i }) => `<option value="${i}"${i === index ? " selected" : ""}>${esc(p.name)}</option>`).join("")}</select>`;
const rounded = (v) => Number(v.toPrecision(4));
const inputs = (container, ...ids) => ids.map((id) => Number(container.querySelector(`#${id}`).value));
const scanDim = (s) => (freeParams(s).length > 1 ? String(App.ui("scan.dim", "1")) : "1");
const dimSwitch = (s) => App.segmented("scan.dim", scanDim(s), [["1", "one parameter"], ["2", "two parameters", freeParams(s).length < 2]]);
const lowerPoint = (below) => (below > 1e-3 ? ` <b>The scan found a point ${num(below, 2)} below the fit result</b>, so the fit stopped in a local minimum; the regions are measured from the lowest point of the scan.` : "");

function renderScan1d(container, state) {
  const s = state.summary;
  const f = state.fit;
  const index = Number(App.ui("scan.index", s.poiIndex >= 0 ? s.poiIndex : freeParams(s)[0].i));
  const p = s.params[index];
  const done = state.scan && state.scan.index === index ? state.scan : null;
  const [lo, hi] = done ? [done.values[0], done.values[done.values.length - 1]] : App.scanRange(index);
  container.innerHTML =
    App.toolbar(
      dimSwitch(s) +
        `<label>parameter ${paramSelect(s, "scan.index", index)}</label>` +
        `<label>from <input id="scan-lo" type="number" step="any" value="${rounded(lo)}"></label>` +
        `<label>to <input id="scan-hi" type="number" step="any" value="${rounded(hi)}"></label>` +
        `<label>points <input id="scan-n" type="number" min="5" max="201" step="1" value="${done ? done.values.length : 25}"></label>` +
        `<label title="Fit each point again from the solutions of its neighbours and keep the lower result. Slower, for likelihoods with several minima."><input id="scan-neighbours" type="checkbox"${(done ? done.neighbours : App.ui("scan.neighbours", false)) ? " checked" : ""}> refit from neighbours</label>` +
        `<button id="scan-run" class="primary" data-job>Scan</button>`
    ) +
    (done ? `<div class="figure" id="scan-figure" style="max-width: 760px"></div><div id="scan-text"></div>` : `<p class="empty">The scan fixes ${esc(p.name)} at each value in turn and minimises over all other parameters.</p>`);
  container.querySelector("#scan-run").onclick = () => {
    const [a, b, n] = inputs(container, "scan-lo", "scan-hi", "scan-n");
    if (a < b && Math.round(n) >= 3) App.actions.scan(index, a, b, Math.round(n), container.querySelector("#scan-neighbours").checked);
  };
  if (!done) return;

  drawScan(container.querySelector("#scan-figure"), done, { name: p.name, best: f.pars[index], unc: f.unc[index] });
  // intervals are quoted around the fit result, or around the lowest scan point if that lies below it
  const centre = done.reference < -1e-3 ? done.interval.min : f.pars[index];
  const end = (v) => (Number.isFinite(v) ? num(v) : "?");
  const side = (v, sign) => (Number.isFinite(v) ? `${sign}${num(Math.abs(v - centre))}` : `${sign}?`);
  const openEnded = done.intervals.some((iv) => iv.ranges.some((r) => !r.every(Number.isFinite)));
  const split = done.intervals.some((iv) => iv.ranges.length > 1);
  // one row per range; the distances to the central value are given for the range that contains it
  const rows = (iv) => {
    const head = (k) => (k ? `<td></td><td></td>` : `<td>${percent(iv.cl)}</td><td class="num">${fmt(iv.level)}</td>`);
    if (!iv.ranges.length) return `<tr>${head(0)}<td class="dim" colspan="3">no scan point inside</td></tr>`;
    return iv.ranges
      .map(([lo, hi], k) => `<tr>${head(k)}<td class="num"><b>${end(lo)}</b></td><td class="num"><b>${end(hi)}</b></td><td class="num">${lo <= centre && centre <= hi ? `${side(lo, "−")} / ${side(hi, "+")}` : ""}</td></tr>`)
      .join("");
  };
  container.querySelector("#scan-text").innerHTML =
    `<div class="results"><table class="data"><tr><th>confidence interval for ${esc(p.name)}</th><th class="num">−2 Δln <i>L</i></th><th class="num">from</th><th class="num">to</th><th class="num">around ${num(centre)}</th></tr>` +
    done.intervals.map(rows).join("") +
    `</table></div>` +
    `<p class="note">The fit gave ${num(f.pars[index])}${f.unc[index] > 0 ? ` ± ${num(f.unc[index], 2)}` : ""}. The ends of the intervals are found with further fits between the scan points.` +
    (split ? ` The likelihood has more than one minimum in the scanned range, so a confidence region can consist of several separate intervals; each is listed on its own row.` : "") +
    (openEnded ? ` A <b>?</b> marks an end outside the scanned range.` : "") +
    lowerPoint(-done.reference) +
    ` ${done.values.length} scan points in ${App.duration(done.seconds)}${done.neighbours ? `, ${done.refits} of them lower when refitted from a neighbour` : ""}.</p>`;
}

function renderScan2d(container, state) {
  const s = state.summary;
  const f = state.fit;
  const [ix, iy] = App.scanPair();
  const done = state.scan2d && state.scan2d.indices[0] === ix && state.scan2d.indices[1] === iy ? state.scan2d : null;
  const names = [s.params[ix].name, s.params[iy].name];
  const axis = (k, index, grid) => {
    const [lo, hi] = grid ? [grid[0], grid[grid.length - 1]] : App.scanRange(index);
    return (
      `<label>${k} ${paramSelect(s, `scan.${k}`, index, k === "x" ? -1 : ix)}</label>` +
      `<label>from <input id="scan-${k}lo" type="number" step="any" value="${rounded(lo)}"></label>` +
      `<label>to <input id="scan-${k}hi" type="number" step="any" value="${rounded(hi)}"></label>` +
      `<label>points <input id="scan-${k}n" type="number" min="5" max="101" step="1" value="${grid ? grid.length : 15}"></label>`
    );
  };
  container.innerHTML =
    App.toolbar(dimSwitch(s) + axis("x", ix, done && done.x)) +
    App.toolbar(axis("y", iy, done && done.y) + `<button id="scan-run" class="primary" data-job>Scan</button>`) +
    (done ? `<div class="figure" id="scan-figure" style="max-width: 640px"></div><p class="note" id="scan-text"></p>` : `<p class="empty">The scan fixes ${esc(names[0])} and ${esc(names[1])} at each point of a grid and minimises over all other parameters: one fit per point.</p>`);
  container.querySelector("#scan-run").onclick = () => {
    const [xa, xb, xn, ya, yb, yn] = inputs(container, "scan-xlo", "scan-xhi", "scan-xn", "scan-ylo", "scan-yhi", "scan-yn");
    if (xa < xb && ya < yb && Math.round(xn) >= 3 && Math.round(yn) >= 3) App.actions.scan2d([ix, iy], [xa, xb, Math.round(xn)], [ya, yb, Math.round(yn)]);
  };
  if (!done) return;

  drawScan2d(container.querySelector("#scan-figure"), done, { names, best: [f.pars[ix], f.pars[iy]] });
  const z = done.twiceDeltaNll.flat();
  const failed = z.filter((v) => !Number.isFinite(v)).length;
  container.querySelector("#scan-text").innerHTML =
    `Regions where −2 Δln <i>L</i> is below ${done.levels.map((l) => `${fmt(l.level)} (${percent(l.cl)} CL)`).join(" and ")}, the values for two parameters. ` +
    `The contours are interpolated between the grid points.` +
    (failed ? ` The likelihood is not defined at ${failed} of the points.` : "") +
    lowerPoint(-Math.min(...z.filter(Number.isFinite))) +
    ` ${z.length} fits in ${App.duration(done.seconds)}.`;
}

App.view({
  id: "scans",
  title: "Scans",
  needsFit: true,
  render: (container, state) => (scanDim(state.summary) === "2" ? renderScan2d : renderScan1d)(container, state),
});

App.view({
  id: "limits",
  title: "Limits",
  needsFit: true,
  needsPoi: true,
  render(container, state) {
    const s = state.summary;
    const lim = state.limit;
    const sig = state.significance;
    const logy = App.ui("limit.log", false);
    const poi = esc(s.poiName);
    container.innerHTML =
      App.toolbar(
        `<button id="limit-run" class="primary" data-job>Compute upper limit</button>` +
          `<label>scan points <select id="limit-n">${[11, 21, 41].map((v) => `<option${v === 21 ? " selected" : ""}>${v}</option>`).join("")}</select></label>` +
          `<label><input type="checkbox" data-ui="limit.log"${logy ? " checked" : ""}> logarithmic</label>` +
          `<span class="spacer"></span><button id="sig-run" data-job>Compute significance</button>`
      ) +
      (lim
        ? `<div class="figure" id="limit-figure" style="max-width: 760px"></div>` +
          `<div class="results"><table class="data"><tr><th>95% CL upper limit on ${poi}</th><th class="num"></th></tr>` +
          `<tr><td>observed</td><td class="num"><b>${num(lim.observed)}</b></td></tr>` +
          `<tr><td>expected (median)</td><td class="num">${num(lim.expected[2])}</td></tr>` +
          `<tr><td>expected ±1σ</td><td class="num">${num(lim.expected[1])} … ${num(lim.expected[3])}</td></tr>` +
          `<tr><td>expected ±2σ</td><td class="num">${num(lim.expected[0])} … ${num(lim.expected[4])}</td></tr></table></div>` +
          `<p class="note">CL<sub>s</sub> with the asymptotic formulae, test statistic ${lim.tilde ? "q̃<sub>μ</sub> (the range of " + poi + " starts at 0)" : "q<sub>μ</sub> (the range of " + poi + " does not start at 0)"}. Expected values are for the background-only hypothesis.` +
          (lim.expected.concat([lim.observed]).some((v) => !Number.isFinite(v)) ? ` <b>Some limits lie outside the range of ${poi}</b>; widen the range above and compute again.` : "") +
          `</p>`
        : `<p class="empty">Scans the signal strength and finds where CL<sub>s</sub> drops below 0.05. Each point takes two fits.</p>`) +
      (sig
        ? `<h2>Discovery significance</h2><div class="results"><table class="data">` +
          `<tr><td>observed</td><td class="num"><b>${fmt(sig.z)} σ</b></td><td class="num">p<sub>0</sub> = ${num(sig.p0, 2)}</td></tr>` +
          `<tr><td>expected for ${poi} = 1</td><td class="num">${fmt(sig.zExpected)} σ</td><td class="num">p<sub>0</sub> = ${num(sig.p0Expected, 2)}</td></tr></table></div>` +
          `<p class="note">From the test statistic q<sub>0</sub> = ${num(sig.q0)} with the asymptotic formulae.</p>`
        : "");
    container.querySelector("#limit-run").onclick = () => App.actions.limit(Number(container.querySelector("#limit-n").value));
    container.querySelector("#sig-run").onclick = () => App.actions.significance();
    if (!lim) return;

    drawLimit(container.querySelector("#limit-figure"), lim, { poiName: s.poiName, logy });
  },
});
