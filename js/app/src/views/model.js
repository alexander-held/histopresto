// Model tab: what is in the workspace.
import { esc, num, yieldFmt } from "@histopresto/viz";
import { App } from "../app.js";

const MAX_ROWS = 300;

App.view({
  id: "model",
  title: "Model",
  render(container, state) {
    const s = state.summary;
    const constrained = s.params.filter((p) => p.constraint !== "none" && !p.fixed).length;
    const fixed = s.params.filter((p) => p.fixed).length;
    const tile = (value, label) => `<div class="tile"><div class="value">${value}</div><div class="label">${label}</div></div>`;
    const types = Object.entries(s.modifierCounts).sort((a, b) => b[1] - a[1]);

    const filter = App.ui("model.filter", "").toLowerCase();
    const kind = App.ui("model.kind", "all");
    const rows = s.params
      .map((p, i) => ({ p, i }))
      .filter(({ p }) => (!filter || p.name.toLowerCase().includes(filter)) && (kind === "all" || (kind === "fixed" ? p.fixed : p.types.includes(kind))));

    container.innerHTML =
      `<div class="tiles">` +
      tile(s.channels.length, s.channels.length === 1 ? "channel" : "channels") +
      tile(s.nBins, "bins") +
      tile(s.samples.length, "samples") +
      tile(s.nPars, "parameters") +
      tile(constrained, "with a constraint term") +
      tile(fixed, "held constant") +
      `</div>` +
      `<p class="note">Measurement <b>${esc(s.measurement)}</b>` +
      (s.poiIndex >= 0 ? `, parameter of interest <span class="mono">${esc(s.poiName)}</span>` : `, no parameter of interest`) +
      `. Modifiers: ${types.map(([t, n]) => `${n} ${t}`).join(", ")}.</p>` +
      `<h2>Channels</h2><div class="table-wrap"><table class="data"><tr><th>channel</th><th class="num">bins</th><th class="num">samples</th><th class="num">data</th><th class="num">pre-fit expectation</th></tr>` +
      s.prefit.channels
        .map((ch) => `<tr><td>${esc(ch.name)}</td><td class="num">${ch.data.length}</td><td class="num">${ch.samples.length}</td><td class="num">${yieldFmt(ch.dataSum)}</td><td class="num">${yieldFmt(ch.sum)} <span class="unc">± ${yieldFmt(ch.sumUnc)}</span></td></tr>`)
        .join("") +
      `</table></div>` +
      `<h2>Parameters</h2>` +
      App.toolbar(
        `<label>filter <input id="model-filter" type="search" data-ui="model.filter" value="${esc(App.ui("model.filter", ""))}" placeholder="part of a name"></label>` +
          `<label>show <select data-ui="model.kind">${["all", ...types.map(([t]) => t), "fixed"].map((t) => `<option${t === kind ? " selected" : ""}>${t}</option>`).join("")}</select></label>` +
          `<span>${rows.length} of ${s.nPars}</span>`
      ) +
      `<div class="table-wrap"><table class="data"><tr><th>parameter</th><th>modifier</th><th>constraint</th><th class="num">initial</th><th class="num">range</th><th class="num">pre-fit uncertainty</th><th></th></tr>` +
      rows
        .slice(0, MAX_ROWS)
        .map(
          ({ p }) =>
            `<tr><td class="mono">${esc(p.name)}</td><td>${p.types.join(", ")}</td><td>${p.constraint === "none" ? '<span class="dim">–</span>' : p.constraint}</td>` +
            `<td class="num">${num(p.init)}</td><td class="num">${num(p.lower)} … ${num(p.upper)}</td><td class="num">${p.prefitUnc > 0 ? num(p.prefitUnc) : "–"}</td><td class="dim">${p.fixed ? "constant" : ""}</td></tr>`
        )
        .join("") +
      `</table></div>` +
      (rows.length > MAX_ROWS ? `<p class="note">Showing the first ${MAX_ROWS}; use the filter to narrow the list.</p>` : "");
  },
});
