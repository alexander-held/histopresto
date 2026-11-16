// Data/MC and Tables tabs: yields before and after the fit.
import { drawYields, esc, fmt, yieldFmt } from "@histopresto/viz";
import { App } from "../app.js";


// which yields to show: post-fit once there is a fit, unless pre-fit was chosen
function chosen(state) {
  const mode = state.fit ? App.ui("yields.mode", "post") : "pre";
  return { mode, y: mode === "post" ? state.fit.postfit : state.summary.prefit };
}
const modeSwitch = (state, mode) => App.segmented("yields.mode", mode, [["pre", "pre-fit"], ["post", "post-fit", !state.fit]]);

// samples of a channel in stacking order with their colours; small backgrounds are merged into "other"
function stack(summary, ch) {
  const layers = [];
  const other = ch.samples.filter((s) => !summary.colours[s.name]);
  if (other.length) {
    layers.push({ name: other.length === 1 ? other[0].name : `other (${other.length})`, colour: "var(--other)", yields: ch.total.map((_, b) => other.reduce((t, s) => t + s.yields[b], 0)) });
  }
  for (const name of summary.stackOrder) {
    const s = ch.samples.find((x) => x.name === name);
    if (s) layers.push({ name, colour: summary.colours[name], yields: s.yields });
  }
  return layers;
}

App.view({
  id: "datamc",
  title: "Data/MC",
  render(container, state) {
    const s = state.summary;
    const { mode, y } = chosen(state);
    const logy = App.ui("yields.log", false);
    const which = App.ui("yields.channel", "all");
    const channels = y.channels.filter((ch) => which === "all" || ch.name === which);
    container.innerHTML =
      App.toolbar(
        modeSwitch(state, mode) +
          `<label><input type="checkbox" data-ui="yields.log"${logy ? " checked" : ""}> logarithmic</label>` +
          (s.channels.length > 1 ? `<label>channel <select data-ui="yields.channel"><option value="all">all</option>${s.channels.map((c) => `<option${c.name === which ? " selected" : ""}>${esc(c.name)}</option>`).join("")}</select></label>` : "")
      ) + `<div class="figures${channels.length === 1 ? " single" : ""}">${channels.map(() => `<div class="figure"></div>`).join("")}</div>` +
      `<p class="note">The hatched band is the ${mode}-fit uncertainty of the total expectation${mode === "post" ? ", propagated from the covariance matrix of the fit" : ", from the widths of the constraint terms (unconstrained parameters do not contribute)"}.</p>`;
    const boxes = container.querySelectorAll(".figure");
    channels.forEach((ch, i) => drawYields(boxes[i], ch, stack(s, ch), { mode, logy, wide: channels.length === 1 }));
  },
});

App.view({
  id: "tables",
  title: "Tables",
  render(container, state) {
    const s = state.summary;
    const { mode, y } = chosen(state);
    // value and uncertainty with the same number of decimals
    const cell = (v, u) => {
      if (v === null) return `<td class="num"><span class="dim">–</span></td>`;
      const d = Math.abs(v) >= 1000 ? 0 : Math.abs(v) >= 10 ? 1 : 2;
      return `<td class="num">${fmt(v, d)} <span class="unc">± ${fmt(u, d)}</span></td>`;
    };
    const head = `<tr><th>sample</th>${y.channels.map((ch) => `<th class="num">${esc(ch.name)}</th>`).join("")}</tr>`;
    const rows = s.samples
      .map((name) => `<tr><td>${esc(name)}</td>${y.channels.map((ch) => { const smp = ch.samples.find((x) => x.name === name); return smp ? cell(smp.sum, smp.sumUnc) : cell(null); }).join("")}</tr>`)
      .join("");
    const total = `<tr class="total"><td>total</td>${y.channels.map((ch) => cell(ch.sum, ch.sumUnc)).join("")}</tr>`;
    const data = `<tr><td>data</td>${y.channels.map((ch) => `<td class="num">${yieldFmt(ch.dataSum)}</td>`).join("")}</tr>`;
    container.innerHTML =
      App.toolbar(modeSwitch(state, mode) + `<span class="spacer"></span><button id="csv">Download CSV</button>`) +
      `<div class="table-wrap"><table class="data">${head}${rows}${total}${data}</table></div>` +
      `<p class="note">Event yields summed over the bins of each channel, ${mode}-fit. Uncertainties include the correlations between bins and between samples.</p>`;
    container.querySelector("#csv").onclick = () => {
      const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
      const lines = [["sample", ...y.channels.flatMap((ch) => [ch.name, `${ch.name} uncertainty`])].map(q).join(",")];
      for (const name of s.samples) {
        lines.push([q(name), ...y.channels.flatMap((ch) => { const smp = ch.samples.find((x) => x.name === name); return smp ? [smp.sum, smp.sumUnc] : ["", ""]; })].join(","));
      }
      lines.push([q("total"), ...y.channels.flatMap((ch) => [ch.sum, ch.sumUnc])].join(","));
      lines.push([q("data"), ...y.channels.flatMap((ch) => [ch.dataSum, ""])].join(","));
      App.download(`${state.name.replace(/\.json$/i, "")}_yields_${mode}fit.csv`, lines.join("\n") + "\n", "text/csv");
    };
  },
});
