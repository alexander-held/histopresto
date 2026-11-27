// Runs the fitting jobs, in a Web Worker where possible (pages opened from file:// cannot start one
// in Chrome; the same jobs then run on the main thread, yielding between fits).
import { Model, infer, num, parseWorkspace, combineWorkspaces } from "@histopresto/core";

const inWorker = typeof document === "undefined" && typeof importScripts === "function";

// ----- jobs: all work on one session {model, lik, base} -----
function Session() {
  this.model = null;
  this.lik = null;
  this.base = null;
}
const J = {};

J.load = async function (s, args) {
  // several workspaces are combined into one model
  const parts = args.workspaces.map(({ name, text }) => {
    try {
      return { name, workspace: parseWorkspace(text) };
    } catch (e) {
      throw new Error(args.workspaces.length > 1 ? `${name}: ${e.message}` : e.message);
    }
  });
  const ws = combineWorkspaces(parts);
  const model = new Model(ws, args.measurement || 0, { poiBounds: args.poiBounds || undefined });
  s.model = model;
  // Asimov data: the expectation at the initial parameter values takes the place of the observed data
  if (args.asimov) model.observed = model.expectedData(model.init);
  s.lik = model.likelihood(model.observed);
  s.base = null;
  const kinds = { normal: "Gaussian", poisson: "Poisson", free: "none" };
  const modifierCounts = {};
  for (const set of model.sets) for (const t of set.types) modifierCounts[t] = (modifierCounts[t] || 0) + 1;
  return {
    notes: ws.notes,
    measurement: model.measurement,
    measurements: ws.measurements.map((m) => m.name),
    channels: model.channels.map((c) => ({ name: c.name, nBins: c.nBins })),
    samples: model.samples,
    nBins: model.nBins,
    nPars: model.nPars,
    nFree: model.fixed.reduce((n, f) => n + (f ? 0 : 1), 0),
    poiIndex: model.poiIndex,
    poiName: model.poiName,
    poiBounds: model.poiIndex >= 0 ? [model.lower[model.poiIndex], model.upper[model.poiIndex]] : null,
    modifierCounts,
    // samples scaled by the POI, to draw them as signal
    signalSamples: [...new Set(model.blocks.filter((b) => b.scal.includes(model.poiIndex) || b.fnPars.includes(model.poiIndex)).map((b) => model.samples[b.sample]))],
    params: model.names.map((name, p) => {
      const set = model.sets[model.setIndex[p]];
      return {
        name,
        set: set.name,
        types: set.types,
        constraint: kinds[set.kind],
        init: model.init[p],
        lower: model.lower[p],
        upper: model.upper[p],
        fixed: !!model.fixed[p],
        prefitUnc: model.prefitUnc[p],
        aux: model.auxIndex[p] >= 0 ? model.auxdata[model.auxIndex[p]] : null,
      };
    }),
    prefit: infer.yields(model, model.init, null),
  };
};

// time spent computing, without the pauses in which progress is reported
function stopwatch(ctx) {
  const start = performance.now();
  let paused = 0;
  const tick = async (info) => {
    const t = performance.now();
    await ctx.tick(info);
    paused += performance.now() - t;
  };
  return { tick, seconds: () => (performance.now() - start - paused) / 1000 };
}

J.fit = async function (s, args, ctx) {
  const watch = stopwatch(ctx);
  const base = await infer.fit(s.model, s.lik, {}, watch);
  const seconds = watch.seconds();
  s.base = base;
  await ctx.tick({ label: "post-fit yields" });
  return {
    seconds,
    pars: base.pars,
    unc: base.unc,
    twiceNll: base.twiceNll,
    status: base.status,
    atBound: base.atBound,
    inner: base.inner,
    correlation: base.cov ? infer.correlation(base) : null,
    postfit: infer.yields(s.model, base.pars, base.cov ? base : null),
    impacts: s.model.poiIndex >= 0 && base.cov && base.pos[s.model.poiIndex] >= 0 ? infer.approximateImpacts(s.model, base) : [],
  };
};

J.ranking = async (s, args, ctx) => infer.ranking(s.model, s.lik, s.base, args.indices, ctx);
J.scan = async (s, args, ctx) => {
  const watch = stopwatch(ctx);
  const r = await infer.scan(s.model, s.lik, s.base, args.index, num.linspace(args.lo, args.hi, args.points), { neighbours: args.neighbours }, watch);
  return { ...r, neighbours: args.neighbours, seconds: watch.seconds() };
};
J.scan2d = async (s, args, ctx) => {
  const watch = stopwatch(ctx);
  const r = await infer.scan2d(s.model, s.lik, s.base, args.indices, num.linspace(...args.x), num.linspace(...args.y), watch);
  return { ...r, seconds: watch.seconds() };
};
J.limit = async (s, args, ctx) => infer.limit(s.model, s.lik, s.base, args, ctx);
J.significance = async (s, args, ctx) => infer.significance(s.model, s.lik, s.base, ctx);

const CANCELLED = "cancelled";
const pause = () => new Promise((resolve) => setTimeout(resolve, 0));

if (inWorker) {
  const session = new Session();
  let cancel = false;
  self.onmessage = async (event) => {
    const { id, job, args } = event.data;
    if (job === "cancel") {
      cancel = true;
      return;
    }
    cancel = false;
    const ctx = {
      tick: async (info) => {
        self.postMessage({ id, progress: info });
        await pause();
        if (cancel) throw new Error(CANCELLED);
      },
    };
    try {
      self.postMessage({ id, result: await J[job](session, args, ctx) });
    } catch (e) {
      self.postMessage({ id, error: e.message });
    }
  };
}

// ----- main thread side -----
function Engine(workerUrl) {
  this.worker = null;
  this.pending = new Map();
  this.nextId = 1;
  this.session = new Session();
  this.cancelFlag = false;
  this.busy = false;
  try {
    if (location.protocol !== "file:") {
      this.worker = new Worker(workerUrl);
      this.worker.onmessage = (event) => this._receive(event.data);
      this.worker.onerror = (event) => {
        // the worker could not be started: fall back to the main thread for jobs still waiting
        event.preventDefault();
        this.worker = null;
        for (const [id, p] of this.pending) {
          this.pending.delete(id);
          this._local(p.job, p.args, p.onProgress).then(p.resolve, p.reject);
        }
      };
    }
  } catch (e) {
    this.worker = null;
  }
}
Engine.prototype._receive = function (msg) {
  const p = this.pending.get(msg.id);
  if (!p) return;
  if (msg.progress) {
    if (p.onProgress) p.onProgress(msg.progress);
    return;
  }
  this.pending.delete(msg.id);
  if (msg.error !== undefined) p.reject(new Error(msg.error));
  else p.resolve(msg.result);
};
Engine.prototype._local = async function (job, args, onProgress) {
  this.cancelFlag = false;
  const ctx = {
    tick: async (info) => {
      if (onProgress) onProgress(info);
      // let the page repaint; animation frames do not fire in background tabs, hence the timeout
      await new Promise((resolve) => {
        const fallback = setTimeout(resolve, 30);
        requestAnimationFrame(() => {
          clearTimeout(fallback);
          setTimeout(resolve, 0);
        });
      });
      if (this.cancelFlag) throw new Error(CANCELLED);
    },
  };
  return J[job](this.session, args, ctx);
};
// run a job; returns a promise for its result
Engine.prototype.run = function (job, args = {}, onProgress) {
  this.busy = true;
  const done = () => (this.busy = false);
  let promise;
  if (this.worker) {
    promise = new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject, onProgress, job, args });
      this.worker.postMessage({ id, job, args });
    });
  } else promise = this._local(job, args, onProgress);
  promise.then(done, done);
  return promise;
};
Engine.prototype.cancel = function () {
  if (this.worker) this.worker.postMessage({ job: "cancel" });
  else this.cancelFlag = true;
};
Engine.prototype.threaded = function () {
  return !!this.worker;
};

export { Engine, CANCELLED };
