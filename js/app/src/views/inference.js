// Ranking, Scans and Limits tabs.
import { drawLimit, drawRanking, drawScan, esc, fmt, num, pull } from "@histopresto/viz";
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

App.view({
  id: "scans",
  title: "Scans",
  needsFit: true,
  render(container, state) {
    const s = state.summary;
    const f = state.fit;
    const candidates = s.params.map((p, i) => ({ p, i })).filter(({ p }) => !p.fixed);
    const index = Number(App.ui("scan.index", s.poiIndex >= 0 ? s.poiIndex : candidates[0].i));
    const p = s.params[index];
    const done = state.scan && state.scan.index === index ? state.scan : null;
    const [lo, hi] = done ? [done.values[0], done.values[done.values.length - 1]] : App.scanRange(index);
    container.innerHTML =
      App.toolbar(
        `<label>parameter <select data-ui="scan.index" data-num style="max-width: 22rem">${candidates.map(({ p: q, i }) => `<option value="${i}"${i === index ? " selected" : ""}>${esc(q.name)}</option>`).join("")}</select></label>` +
          `<label>from <input id="scan-lo" type="number" step="any" value="${Number(lo.toPrecision(4))}"></label>` +
          `<label>to <input id="scan-hi" type="number" step="any" value="${Number(hi.toPrecision(4))}"></label>` +
          `<label>points <input id="scan-n" type="number" min="5" max="201" step="1" value="${done ? done.values.length : 25}"></label>` +
          `<button id="scan-run" class="primary" data-job>Scan</button>`
      ) +
      (done ? `<div class="figure" id="scan-figure" style="max-width: 760px"></div><div id="scan-text" class="note"></div>` : `<p class="empty">The scan fixes ${esc(p.name)} at each value in turn and minimises over all other parameters.</p>`);
    container.querySelector("#scan-run").onclick = () => {
      const a = Number(container.querySelector("#scan-lo").value);
      const b = Number(container.querySelector("#scan-hi").value);
      const points = Math.round(Number(container.querySelector("#scan-n").value));
      if (a < b && points >= 3) App.actions.scan(index, a, b, points);
    };
    if (!done) return;

    const y = done.twiceDeltaNll;
    drawScan(container.querySelector("#scan-figure"), done, { name: p.name, best: f.pars[index], unc: f.unc[index] });
    const iv = done.interval;
    const side = (v, sign) => (Number.isFinite(v) ? `${sign}${num(Math.abs(v - iv.min), 2)}` : `${sign}? (outside the scanned range)`);
    container.querySelector("#scan-text").innerHTML =
      `Minimum of the scan at <b>${num(iv.min)}</b>, interval where −2 Δln <i>L</i> &lt; 1: <b>${side(iv.lo, "−")} / ${side(iv.hi, "+")}</b>. ` +
      `The fit gave ${num(f.pars[index])}${f.unc[index] > 0 ? ` ± ${num(f.unc[index], 2)}` : ""}.` +
      (Math.min(...y) < -1e-3 ? ` <b>The scan found a point ${num(-Math.min(...y), 2)} below the fit result</b>, so the fit stopped in a local minimum.` : "");
  },
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
