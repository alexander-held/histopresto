// Page logic: loading a workspace, running jobs, tabs. The views register themselves with App.view().
import { esc, fmt, num, yieldFmt, layoutKey } from "@histopresto/viz";
import { Engine, CANCELLED } from "./engine.js";

const $ = (id) => document.getElementById(id);
const duration = (s) => (s < 1 ? `${Math.max(1, Math.round(1000 * s))} ms` : `${s.toFixed(s < 10 ? 2 : 1)} s`);
const SLOTS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)"];

const App = {
  $, esc, fmt, num, yieldFmt, duration,
  views: [],
  state: null,
  current: "model",
  busy: false,
  view(v) {
    this.views.push(v);
  },
};

// sources: [{name, text}], the workspaces that are combined into the model
function freshState(sources) {
  return { name: sources.map((s) => s.name).join(" + "), sources, measurement: 0, poiBounds: null, asimov: false, summary: null, fit: null, exact: new Map(), scan: null, scan2d: null, limit: null, significance: null, ui: { ...App.presets } };
}

// ----- errors, progress -----
function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function setBusy(on, label) {
  App.busy = on;
  $("progress").hidden = !on;
  $("fit").disabled = on;
  for (const b of document.querySelectorAll("[data-job]")) b.disabled = on;
  if (on) progress({ label });
}
function progress(info) {
  const fill = $("progress-fill");
  if (info.total) {
    fill.classList.remove("indeterminate");
    fill.style.width = `${Math.round((100 * (info.done + 1)) / info.total)}%`;
    fill.style.marginLeft = "0";
  } else {
    fill.classList.add("indeterminate");
    fill.style.width = "30%";
  }
  $("progress-label").textContent = (info.label || "") + (info.total ? `  (${info.done + 1} / ${info.total})` : "");
}
// run a job in the engine; resolves with the result, or null if it failed or was cancelled
App.run = async function (job, args, label) {
  if (App.busy) return null;
  showError("");
  setBusy(true, label);
  try {
    return await App.engine.run(job, args, progress);
  } catch (e) {
    if (e.message !== CANCELLED) showError(e.message);
    return null;
  } finally {
    setBusy(false);
  }
};

// ----- loading -----
async function load(sources) {
  await open(freshState(sources));
}
async function open(state) {
  const summary = await App.run("load", { workspaces: state.sources, measurement: state.measurement, poiBounds: state.poiBounds, asimov: state.asimov }, "reading the workspace");
  if (!summary) return;
  state.summary = summary;
  App.state = state;
  assignColours(summary);
  $("loader").classList.add("compact");
  const several = state.sources.length > 1;
  $("files").innerHTML = state.sources
    .map((src, i) => `<span class="file">${esc(src.name)}${several ? ` <button class="quiet" data-remove="${i}" title="Remove this workspace" aria-label="Remove ${esc(src.name)}">×</button>` : ""}</span>`)
    .join("");
  $("file-info").textContent = `${summary.channels.length} channel${summary.channels.length === 1 ? "" : "s"}, ${summary.nBins} bins, ${summary.samples.length} samples, ${summary.nPars} parameters`;
  $("workspace").hidden = false;
  const sel = $("measurement");
  sel.innerHTML = summary.measurements.map((m, i) => `<option value="${i}">${esc(m)}</option>`).join("");
  sel.value = String(state.measurement);
  $("measurement-label").hidden = summary.measurements.length < 2;
  $("poi-label").hidden = summary.poiIndex < 0;
  if (summary.poiIndex >= 0) {
    $("poi-name").textContent = summary.poiName;
    $("poi-lo").value = summary.poiBounds[0];
    $("poi-hi").value = summary.poiBounds[1];
  }
  $("notes").innerHTML = summary.notes.map((n) => `<p class="note warn">${esc(n)}</p>`).join("");
  $("dataset").value = state.asimov ? "asimov" : "observed";
  $("download").disabled = true;
  $("status").innerHTML = "";
  if (App.views.find((v) => v.id === App.current).needsPoi && summary.poiIndex < 0) App.current = "model";
  renderTabs();
  render();
  await fit();
  if (App.state === state && state.fit && pendingActions.length) {
    const todo = pendingActions.splice(0);
    for (const name of todo) {
      const usable = name in App.actions && (name.startsWith("scan") || (state.summary.poiIndex >= 0 && (name !== "ranking" || state.fit.impacts.length)));
      if (usable) await App.actions[name]();
    }
  }
}
const pendingActions = (new URLSearchParams(location.search).get("run") || "").split(",").filter(Boolean);
function assignColours(summary) {
  // colour follows the sample: backgrounds by size, signal in red, everything past seven backgrounds is "other"
  const totals = {};
  for (const ch of summary.prefit.channels) for (const s of ch.samples) totals[s.name] = (totals[s.name] || 0) + s.sum;
  const signal = summary.signalSamples[0];
  const backgrounds = summary.samples.filter((s) => s !== signal).sort((a, b) => totals[b] - totals[a]);
  const colours = {};
  backgrounds.forEach((s, i) => (colours[s] = i < SLOTS.length ? SLOTS[i] : null));
  if (signal) colours[signal] = "var(--s8)";
  summary.colours = colours;
  summary.stackOrder = backgrounds.filter((s) => colours[s]).concat(signal ? [signal] : []); // bottom to top, "other" below
}

async function fit() {
  const state = App.state;
  const result = await App.run("fit", {}, "fitting");
  if (!result || state !== App.state) return;
  state.fit = result;
  state.exact = new Map();
  state.scan = state.scan2d = state.limit = state.significance = null;
  const s = state.summary;
  const st = result.status;
  let html;
  if (!Number.isFinite(result.twiceNll)) {
    html = `<span class="warn">The likelihood is not defined at the starting values: a bin with data has an expectation of zero or below.</span>`;
  } else {
    html = st.converged ? `<span class="ok">Fit converged</span>` : `<span class="warn">${st.posDef ? "Fit did not converge" : "The data do not determine all parameters (covariance matrix not positive definite); uncertainties are unreliable"}</span>`;
    html += ` · −2 ln <i>L</i> = <b>${fmt(result.twiceNll, 2)}</b> · fit time <b>${duration(result.seconds)}</b>`;
  }
  if (state.asimov) html = `<b>Asimov data</b> · ${html}`;
  if (s.poiIndex >= 0) {
    const p = s.poiIndex;
    html += ` · ${esc(s.poiName)} = <b>${num(result.pars[p])}${result.atBound[p] ? "</b> (at the edge of its range)" : ` ± ${num(result.unc[p], 2)}</b>`}`;
  }
  $("status").innerHTML = html;
  $("download").disabled = false;
  renderTabs();
  render();
}

// ----- jobs started from the views (or from the address bar: ?run=limit,significance,ranking,scan) -----
App.scanRange = function (index) {
  const { summary: s, fit: f } = App.state;
  const p = s.params[index];
  const sigma = f.unc[index] > 0 ? f.unc[index] : (p.upper - p.lower) / 10;
  return [Math.max(p.lower, f.pars[index] - 3 * sigma), Math.min(p.upper, f.pars[index] + 3 * sigma)];
};
// the two parameters of a 2D scan: as chosen, by default the POI (or the first parameter) and the one most correlated with it
App.scanPair = function () {
  const { summary: s, fit: f } = App.state;
  const free = s.params.map((p, i) => i).filter((i) => !s.params[i].fixed);
  if (free.length < 2) return null;
  const x = Number(App.ui("scan.x", s.poiIndex >= 0 && free.includes(s.poiIndex) ? s.poiIndex : free[0]));
  const k = f.inner.length;
  const a = f.inner.indexOf(x);
  let partner = free.find((i) => i !== x);
  if (f.correlation && a >= 0) {
    let best = -1;
    f.inner.forEach((i, b) => {
      if (b !== a && Math.abs(f.correlation[a * k + b]) > best) [best, partner] = [Math.abs(f.correlation[a * k + b]), i];
    });
  }
  const y = Number(App.ui("scan.y", partner));
  return [x, y === x ? partner : y];
};
App.rankingCandidates = () => App.state.fit.impacts.slice(0, Math.min(App.ui("rank.n", 15), App.state.fit.impacts.length));
async function job(name, args, label, store) {
  const state = App.state;
  const result = await App.run(name, args, label);
  if (!result || state !== App.state) return;
  store(state, result);
  render();
}
App.actions = {
  limit: (points = 21) => job("limit", { points }, "limit", (st, r) => (st.limit = r)),
  significance: () => job("significance", {}, "significance", (st, r) => (st.significance = r)),
  ranking: (indices) =>
    job("ranking", { indices: indices || App.rankingCandidates().map((r) => r.index).filter((i) => !App.state.exact.has(i)) }, "ranking", (st, r) => {
      for (const row of r) st.exact.set(row.index, row);
    }),
  scan: (index, lo, hi, points = 25, neighbours = Boolean(App.ui("scan.neighbours", false))) => {
    const s = App.state.summary;
    if (index === undefined) index = Number(App.ui("scan.index", s.poiIndex >= 0 ? s.poiIndex : s.params.findIndex((p) => !p.fixed)));
    if (lo === undefined) [lo, hi] = App.scanRange(index);
    return job("scan", { index, lo, hi, points, neighbours }, "scan", (st, r) => (st.scan = r));
  },
  // x, y: [from, to, points] for the two parameters
  scan2d: (indices, x, y) => {
    if (!indices) indices = App.scanPair();
    if (!indices) return null;
    const range = (i, r) => r || [...App.scanRange(i), 15];
    return job("scan2d", { indices, x: range(indices[0], x), y: range(indices[1], y) }, "scan", (st, r) => (st.scan2d = r));
  },
};

// ----- tabs and views -----
const available = (v) => !!App.state && (!v.needsFit || !!App.state.fit) && (!v.needsPoi || App.state.summary.poiIndex >= 0);
function renderTabs() {
  $("tabs").innerHTML = App.views
    .map((v) => `<button role="tab" data-view="${v.id}" aria-selected="${v.id === App.current}"${available(v) ? "" : " disabled"}>${v.title}</button>`)
    .join("");
}
function render() {
  const view = App.views.find((v) => v.id === App.current);
  const container = $("view");
  container.innerHTML = "";
  if (!available(view)) {
    container.innerHTML = `<p class="empty">${view.needsPoi && App.state.summary.poiIndex < 0 ? "This needs a parameter of interest, and the measurement does not define one." : "Available once the fit has finished."}</p>`;
    return;
  }
  view.render(container, App.state);
  for (const b of container.querySelectorAll("[data-job]")) b.disabled = App.busy;
}
App.render = render;

// small helpers for the views
App.toolbar = (html) => `<div class="toolbar">${html}</div>`;
App.ui = (key, fallback) => (key in App.state.ui ? App.state.ui[key] : fallback);
App.setUi = (key, value) => {
  App.state.ui[key] = value;
  render();
};
// delegate change / click events of controls carrying data-ui to the ui state
function wireView() {
  const view = $("view");
  view.addEventListener("change", (e) => {
    const key = e.target.dataset && e.target.dataset.ui;
    if (!key) return;
    const t = e.target;
    const focus = t.id;
    App.setUi(key, t.type === "checkbox" ? t.checked : t.type === "number" || "num" in t.dataset ? Number(t.value) : t.value);
    if (focus && $(focus)) $(focus).focus();
  });
  view.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-ui]");
    if (b) App.setUi(b.dataset.ui, b.dataset.value);
  });
}
App.segmented = (key, value, options) =>
  `<span class="segmented">${options.map(([v, label, disabled]) => `<button data-ui="${key}" data-value="${v}" aria-pressed="${v === value}"${disabled ? " disabled" : ""}>${label}</button>`).join("")}</span>`;

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
App.download = download;

function results() {
  const { summary: s, fit: f, limit, significance, exact } = App.state;
  const k = f.inner.length;
  const out = {
    workspace: App.state.name,
    warnings: s.notes,
    measurement: s.measurement,
    data: App.state.asimov ? "asimov" : "observed",
    fit_seconds: f.seconds,
    twice_nll: f.twiceNll,
    converged: f.status.converged,
    parameters: s.params.map((p, i) => ({ name: p.name, value: f.pars[i], uncertainty: p.fixed || f.atBound[i] ? null : f.unc[i], fixed: p.fixed, at_bound: !!f.atBound[i] })),
    correlation: f.correlation && {
      parameters: Array.from(f.inner, (i) => s.params[i].name),
      matrix: Array.from({ length: k }, (_, a) => Array.from(f.correlation.subarray(a * k, (a + 1) * k))),
    },
    yields: { prefit: s.prefit, postfit: f.postfit },
  };
  if (exact.size) out.ranking = [...exact.values()].map((r) => ({ parameter: s.params[r.index].name, postfit_up: r.postfitUp, postfit_down: r.postfitDown, prefit_up: r.prefitUp, prefit_down: r.prefitDown }));
  if (limit) out.upper_limit = { confidence_level: 1 - limit.level, observed: limit.observed, expected: { "-2": limit.expected[0], "-1": limit.expected[1], "0": limit.expected[2], "+1": limit.expected[3], "+2": limit.expected[4] }, scan: limit.values, cls_observed: limit.results.map((r) => r.cls) };
  if (significance) out.significance = { observed: significance.z, expected: significance.zExpected, p0: significance.p0 };
  return out;
}

// ----- start -----
App.start = function () {
  App.engine = new Engine("worker.js");
  wireView();
  const loader = $("loader");
  // several files are combined; with add they join the workspaces that are loaded already
  const readFiles = async (files, add) => {
    if (!files.length || App.busy) return;
    let sources;
    try {
      sources = await Promise.all(Array.from(files, async (file) => ({ name: file.name, text: await file.text() })));
    } catch (e) {
      return showError("Could not read the files.");
    }
    load(add && App.state ? [...App.state.sources, ...sources] : sources);
  };
  let adding = false;
  const choose = (add) => {
    adding = add;
    $("file").click();
  };
  $("choose").onclick = () => choose(false);
  $("add").onclick = () => choose(true);
  $("file").onchange = (e) => {
    readFiles(Array.from(e.target.files), adding);
    e.target.value = "";
  };
  $("files").onclick = (e) => {
    const i = e.target.dataset.remove;
    if (i !== undefined && !App.busy) load(App.state.sources.filter((_, j) => j !== Number(i)));
  };
  $("paste-toggle").onclick = () => {
    $("paste").hidden = !$("paste").hidden;
    if (!$("paste").hidden) $("paste-text").focus();
  };
  $("paste-load").onclick = () => {
    if ($("paste-text").value.trim()) load([{ name: "pasted workspace", text: $("paste-text").value }]);
  };
  $("another").onclick = () => {
    loader.classList.remove("compact");
    $("workspace").hidden = true;
    showError("");
  };
  for (const type of ["dragenter", "dragover"]) {
    document.addEventListener(type, (e) => {
      e.preventDefault();
      loader.classList.add("over");
    });
  }
  for (const type of ["dragleave", "drop"]) document.addEventListener(type, (e) => (e.preventDefault(), loader.classList.remove("over")));
  document.addEventListener("drop", (e) => readFiles(Array.from(e.dataTransfer.files), false));

  // examples are separate scripts, loaded on demand (this also works from file://)
  const pendingExamples = {};
  window.HF_EXAMPLE = (id, workspace) => pendingExamples[id] && pendingExamples[id](workspace);
  $("examples").innerHTML = (window.HF_EXAMPLES || []).map((ex) => `<button data-example="${ex.id}" title="${esc(ex.note)}">${esc(ex.title)}</button>`).join("");
  $("examples").onclick = (e) => {
    const id = e.target.dataset.example;
    if (!id) return;
    const ex = window.HF_EXAMPLES.find((x) => x.id === id);
    pendingExamples[id] = (workspace) => load([{ name: ex.title, text: JSON.stringify(workspace) }]);
    const script = document.createElement("script");
    script.src = `examples/${id}.js`;
    script.onerror = () => showError(`Could not load the example "${ex.title}".`);
    document.head.appendChild(script);
    script.onload = () => script.remove();
  };

  $("tabs").onclick = (e) => {
    const id = e.target.dataset.view;
    if (!id || e.target.disabled) return;
    App.current = id;
    renderTabs();
    render();
  };
  $("fit").onclick = fit;
  $("cancel").onclick = () => App.engine.cancel();
  // load the workspace again with one of the settings changed
  const reopen = (change) => {
    const { sources, measurement, poiBounds, asimov, ui } = App.state;
    open(Object.assign(freshState(sources), { measurement, poiBounds, asimov, ui }, change));
  };
  $("measurement").onchange = (e) => reopen({ measurement: Number(e.target.value), poiBounds: null });
  $("dataset").onchange = (e) => reopen({ asimov: e.target.value === "asimov" });
  const rebound = () => {
    const lo = Number($("poi-lo").value);
    const hi = Number($("poi-hi").value);
    if (!(lo < hi) || $("poi-lo").value === "" || $("poi-hi").value === "") return showError("The range of the parameter of interest needs a lower bound below the upper bound.");
    reopen({ poiBounds: [lo, hi] });
  };
  $("poi-lo").onchange = rebound;
  $("poi-hi").onchange = rebound;
  $("download").onclick = () => App.download(`${App.state.sources.length > 1 ? "combined" : App.state.name.replace(/\.json$/i, "")}_fit_results.json`, JSON.stringify(results(), (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v), 1), "application/json");

  // redraw figures when the layout changes between narrow and wide
  let key = null;
  window.addEventListener("resize", () => {
    const k = layoutKey($("view"));
    if (k !== key && App.state) {
      key = k;
      render();
    }
  });

  const query = new URLSearchParams(location.search);
  if (App.views.some((v) => v.id === query.get("tab"))) App.current = query.get("tab");
  // any other parameter presets a control of a view, e.g. ?yields.log=true&rank.n=30
  App.presets = {};
  for (const [key, value] of query) {
    if (key.includes(".")) App.presets[key] = value === "true" ? true : value === "false" ? false : Number.isNaN(Number(value)) ? value : Number(value);
  }
  const example = document.querySelector(`[data-example="${CSS.escape(query.get("example") || "")}"]`);
  if (example) example.click();
};

export { App };
