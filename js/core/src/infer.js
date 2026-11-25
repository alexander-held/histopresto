// Statistical inference with a HistFactory model: fits, scans, ranking, limits, significance, yields.
// Long-running functions are async and call ctx.tick({done, total, label}) between fits.
import { normSf, cholesky, cholInverse, inverse, linspace, interval } from "./numerics.js";
import { minimize, hessian } from "./minimize.js";
const NO_CTX = { tick: async () => {} };

const diagGuess = (model, p) => {
  const w = model.prefitUnc[p] > 0 ? model.prefitUnc[p] : Math.max((model.upper[p] - model.lower[p]) / 10, 1e-3);
  return 0.5 * w * w;
};
const stepSize = (model, x, p) => 1e-4 * (model.prefitUnc[p] > 0 ? model.prefitUnc[p] : Math.max(Math.abs(x[p]), 1));

// Maximum likelihood fit with covariance. options.fixed: {index: value} to hold parameters constant.
async function fit(model, lik, options = {}, ctx = NO_CTX) {
  const N = model.nPars;
  const fg = lik.value;
  const x0 = Float64Array.from(options.init || model.init);
  const isFixed = Uint8Array.from(model.fixed);
  for (const [p, v] of Object.entries(options.fixed || {})) {
    isFixed[p] = 1;
    x0[p] = v;
  }
  const free = [];
  for (let p = 0; p < N; p++) if (!isFixed[p]) free.push(p);
  const n = free.length;
  for (const p of free) x0[p] = Math.min(Math.max(x0[p], model.lower[p]), model.upper[p]);

  // start from the constraint widths: the second derivatives at the initial point are rarely positive definite
  // and cost more likelihood calls than they save
  await ctx.tick({ label: "minimising" });
  const diag = Float64Array.from(free, (p) => diagGuess(model, p));
  const res = minimize(fg, x0, model.lower, model.upper, free, { diag, tol: 1e-6 });
  let x = res.x;
  let f = res.f;

  // covariance from the second derivatives at the minimum, for parameters not stuck at a bound
  await ctx.tick({ label: "covariance matrix" });
  const g = new Float64Array(N);
  const atBound = new Uint8Array(N);
  let inner = [];
  let cov = null;
  let posDef = false;
  let edm = res.edm;
  for (let pass = 0; pass < 3 && n > 0; pass++) {
    f = fg(x, g);
    inner = [];
    atBound.fill(0);
    for (const p of free) {
      if ((x[p] <= model.lower[p] && g[p] > 0) || (x[p] >= model.upper[p] && g[p] < 0)) atBound[p] = 1;
      else inner.push(p);
    }
    const k = inner.length;
    const H = hessian(fg, x, inner, inner.map((p) => stepSize(model, x, p)), true);
    const L = cholesky(H, k);
    posDef = !!L;
    const Hinv = L ? cholInverse(L, k) : inverse(H, k);
    if (!Hinv) break;
    cov = new Float64Array(k * k);
    for (let i = 0; i < k * k; i++) cov[i] = 2 * Hinv[i];
    if (!posDef) break;
    // a few Newton steps with this matrix to land exactly on the minimum
    let moved = 0;
    for (let it = 0; it < 12; it++) {
      const step = new Float64Array(k);
      edm = 0;
      for (let a = 0; a < k; a++) {
        let t = 0;
        for (let b = 0; b < k; b++) t += Hinv[a * k + b] * g[inner[b]];
        step[a] = -t;
        edm -= 0.5 * g[inner[a]] * step[a];
      }
      if (edm < 1e-13) break;
      const trial = Float64Array.from(x);
      const gt = new Float64Array(N);
      let scale = 1;
      let ft = Infinity;
      for (let h = 0; h < 8; h++) {
        for (let a = 0; a < k; a++) {
          const p = inner[a];
          trial[p] = Math.min(Math.max(x[p] + scale * step[a], model.lower[p]), model.upper[p]);
        }
        ft = fg(trial, gt);
        if (ft <= f + 1e-12 * Math.abs(f)) break;
        scale *= 0.5;
      }
      if (!(ft <= f + 1e-12 * Math.abs(f))) break;
      for (let a = 0; a < k; a++) {
        const p = inner[a];
        moved = Math.max(moved, Math.abs(trial[p] - x[p]) / Math.sqrt(Math.max(cov[a * k + a], 1e-300)));
      }
      x = trial;
      f = ft;
      g.set(gt);
    }
    if (moved < 0.02) break;
  }

  const pos = new Int32Array(N).fill(-1);
  inner.forEach((p, a) => (pos[p] = a));
  const unc = new Float64Array(N);
  if (cov) inner.forEach((p, a) => (unc[p] = Math.sqrt(Math.max(cov[a * inner.length + a], 0))));
  return {
    pars: x,
    twiceNll: f,
    unc,
    cov,
    inner: Int32Array.from(inner),
    pos,
    free: Int32Array.from(free),
    atBound,
    status: { converged: (res.converged || edm < 1e-4) && posDef && Number.isFinite(f), posDef, edm, iterations: res.iterations },
    cache: new Map(),
  };
}

const covAt = (base, p, q) => (base.pos[p] < 0 || base.pos[q] < 0 ? 0 : base.cov[base.pos[p] * base.inner.length + base.pos[q]]);

// Fit with parameters held at given values (one index and value, or arrays of them), starting from
// an earlier fit and its covariance. init: parameters to start from instead of the linear prediction
// from the covariance.
function conditional(model, lik, base, index, value, init) {
  const held = [].concat(index);
  const values = [].concat(value);
  const free = Array.from(base.free).filter((p) => !held.includes(p));
  const n = free.length;
  const k = base.inner.length;
  const key = held.join();
  let start = base.cache.get(key);
  if (!start) {
    // held parameters with an entry in the covariance matrix, and the inverse of their block of it
    let known = base.cov ? held.filter((p) => covAt(base, p, p) > 0) : [];
    let Sinv = inverse(Float64Array.from(known.flatMap((p) => known.map((q) => covAt(base, p, q)))), known.length);
    if (!Sinv) known = [];
    const m = known.length;
    const at = known.map((p) => base.pos[p]);
    const where = free.map((p) => base.pos[p]);
    // gain: shift of the free parameters per unit shift of the held ones; C: covariance of the free ones given the held ones
    const gain = new Float64Array(n * m);
    const C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      if (!base.cov || where[i] < 0) {
        C[i * n + i] = diagGuess(model, free[i]);
        continue;
      }
      const ri = where[i] * k;
      for (let b = 0; b < m; b++) {
        let t = 0;
        for (let c = 0; c < m; c++) t += base.cov[ri + at[c]] * Sinv[c * m + b];
        gain[i * m + b] = t;
      }
      for (let j = 0; j < n; j++) {
        if (where[j] < 0) continue;
        let v = base.cov[ri + where[j]];
        for (let b = 0; b < m; b++) v -= gain[i * m + b] * base.cov[at[b] * k + where[j]];
        C[i * n + j] = 0.5 * v;
      }
    }
    start = { C, gain, known };
    base.cache.set(key, start);
  }
  const hold = (x) => (held.forEach((p, c) => (x[p] = values[c])), x);
  // linear prediction of where the other parameters go
  const x0 = hold(Float64Array.from(init || base.pars));
  const m = init ? 0 : start.known.length;
  const shift = start.known.map((p) => values[held.indexOf(p)] - base.pars[p]);
  for (let i = 0; i < n && m > 0; i++) {
    const p = free[i];
    if (base.pos[p] < 0) continue;
    let t = base.pars[p];
    for (let b = 0; b < m; b++) t += start.gain[i * m + b] * shift[b];
    x0[p] = Math.min(Math.max(t, model.lower[p]), model.upper[p]);
  }
  let res = minimize(lik.value, x0, model.lower, model.upper, free, { C: start.C, tol: 1e-8 });
  if (!Number.isFinite(res.f)) {
    // the prediction left the region where the model is defined: start from the base fit instead
    res = minimize(lik.value, hold(Float64Array.from(base.pars)), model.lower, model.upper, free, { C: start.C, tol: 1e-8 });
  }
  return { pars: res.x, twiceNll: res.f, converged: res.converged };
}

// -2 Δln L at the edge of the 68.3% (one standard deviation) and 95% confidence regions, for one and for two parameters
const LEVELS = {
  1: [{ cl: 0.6827, level: 1 }, { cl: 0.95, level: 3.841458820694124 }],
  2: [{ cl: 0.6827, level: 2.295748928898636 }, { cl: 0.95, level: 5.991464547107979 }],
};

// Profile likelihood scan of one parameter. intervals: for each confidence level {cl, level, ranges},
// where ranges lists the [from, to] stretches of the scan below the level (more than one if the
// likelihood has several minima), with the ends refined by further fits. An end is infinite if the
// curve is still below the level at the end of the scan.
// A fit can stop in a local minimum that the fit of a neighbouring point avoided. With neighbours,
// each point is fitted again from the solutions of its neighbours, going up and then down, and the
// lower result is kept: three fits per point instead of one. refits: number of points that ended
// lower than their first fit.
async function scan(model, lik, base, index, values, { neighbours = false } = {}, ctx = NO_CTX) {
  // the fit started from pars if it ends below the result r, else r
  const lower = (r, pars, value) => {
    const t = conditional(model, lik, base, index, value, pars);
    return t.twiceNll < r.twiceNll ? t : r;
  };
  const fits = [];
  for (let i = 0; i < values.length; i++) {
    await ctx.tick({ done: i, total: values.length, label: `scan of ${model.names[index]}` });
    fits.push(conditional(model, lik, base, index, values[i]));
  }
  const first = fits.map((r) => r.twiceNll);
  for (const step of neighbours ? [1, -1] : []) {
    for (let i = step === 1 ? 1 : values.length - 2; i >= 0 && i < values.length; i += step) {
      await ctx.tick({ done: step === 1 ? i : values.length - 1 - i, total: values.length, label: `scan of ${model.names[index]}: from neighbouring points` });
      fits[i] = lower(fits[i], fits[i - step].pars, values[i]);
    }
  }
  const refits = fits.filter((r, i) => r.twiceNll < first[i] - 1e-6).length;
  const out = fits.map((r) => r.twiceNll - base.twiceNll);
  const iv = interval(values, out, 1);
  // measured from the fit result, or from the lowest point of the scan should that lie below it
  const ref = Math.min(out[iv.index], 0);
  const last = values.length - 1;
  // where the curve crosses the level between scan point k (below it) and its neighbour (above it)
  const crossing = (level, k, dir) => {
    let a = [values[k], out[k] - ref, fits[k].pars];
    let b = [values[k + dir], out[k + dir] - ref, fits[k + dir].pars];
    if (!Number.isFinite(b[1])) return b[0];
    // regula falsi on the square root, which is a straight line for a parabola
    const root = (y) => Math.sqrt(Math.max(y, 0));
    let x = b[0];
    for (let it = 0; it < 8; it++) {
      x = a[0] + ((root(level) - root(a[1])) / (root(b[1]) - root(a[1]))) * (b[0] - a[0]);
      // from the solutions at both ends, which may be different minima
      const r = neighbours ? lower(conditional(model, lik, base, index, x, a[2]), b[2], x) : conditional(model, lik, base, index, x);
      const y = r.twiceNll - base.twiceNll - ref;
      if (!Number.isFinite(y) || Math.abs(y - level) < 1e-4) break;
      if (y < level) a = [x, y, r.pars];
      else b = [x, y, r.pars];
    }
    return x;
  };
  const intervals = [];
  for (const { cl, level } of LEVELS[1]) {
    await ctx.tick({ label: `scan of ${model.names[index]}: ${Number((100 * cl).toFixed(1))}% interval` });
    const below = (k) => out[k] - ref < level;
    const ranges = [];
    for (let k = 0; k <= last; k++) {
      if (!below(k)) continue;
      const first = k;
      while (k < last && below(k + 1)) k++;
      ranges.push([first === 0 ? -Infinity : crossing(level, first, -1), k === last ? Infinity : crossing(level, k, 1)]);
    }
    intervals.push({ cl, level, ranges });
  }
  return { index, values, twiceDeltaNll: out, interval: iv, reference: ref, intervals, refits };
}

// Profile likelihood scan of two parameters on a grid: twiceDeltaNll[i][j] for x[i], y[j].
async function scan2d(model, lik, base, indices, x, y, ctx = NO_CTX) {
  const z = [];
  const total = x.length * y.length;
  const label = `scan of ${model.names[indices[0]]} and ${model.names[indices[1]]}`;
  let last = -Infinity;
  for (let i = 0; i < x.length; i++) {
    const row = [];
    for (let j = 0; j < y.length; j++) {
      // fits can be fast: report progress (and allow cancelling) a few times a second only
      if (Date.now() - last > 80) {
        await ctx.tick({ done: i * y.length + j, total, label });
        last = Date.now();
      }
      row.push(conditional(model, lik, base, indices, [x[i], y[j]]).twiceNll - base.twiceNll);
    }
    z.push(row);
  }
  return { indices, x, y, twiceDeltaNll: z, levels: LEVELS[2] };
}

// Impact of each nuisance parameter on the POI as estimated from the covariance matrix.
function approximateImpacts(model, base) {
  const poi = model.poiIndex;
  const out = [];
  for (const p of base.inner) {
    if (p === poi) continue;
    const sigma = base.unc[p];
    if (!(sigma > 0)) continue;
    const c = covAt(base, poi, p);
    out.push({ index: p, postfit: c / sigma, prefit: (c / (sigma * sigma)) * model.prefitUnc[p], exact: false });
  }
  out.sort((u, v) => Math.abs(v.postfit) - Math.abs(u.postfit));
  return out;
}

// Impacts from fits with the nuisance parameter held at its best-fit value +- uncertainty.
async function ranking(model, lik, base, indices, ctx = NO_CTX) {
  const poi = model.poiIndex;
  const mu = base.pars[poi];
  const out = [];
  for (let i = 0; i < indices.length; i++) {
    const p = indices[i];
    await ctx.tick({ done: i, total: indices.length, label: `ranking: ${model.names[p]}` });
    const shift = (d) => {
      if (!(d > 0)) return [0, 0];
      return [1, -1].map((sign) => conditional(model, lik, base, p, base.pars[p] + sign * d).pars[poi] - mu);
    };
    const post = shift(base.unc[p]);
    const pre = shift(model.prefitUnc[p]);
    out.push({ index: p, postfitUp: post[0], postfitDown: post[1], prefitUp: pre[0], prefitDown: pre[1], exact: true });
  }
  return out;
}

// ----- hypothesis tests with the asymptotic formulae (arXiv:1007.1727) -----
// Returns a function mu -> {cls, clsb, clb, expected[5] for (-2..+2 sigma), q, qAsimov}.
function tester(model, lik, base) {
  const poi = model.poiIndex;
  const tilde = model.lower[poi] === 0;
  const muHat = base.pars[poi];
  const asimovPars = conditional(model, lik, base, poi, 0).pars;
  const asimovLik = model.likelihood(model.expectedData(asimovPars));
  const asimovBase = { ...base, pars: asimovPars, twiceNll: asimovLik.value(asimovPars), cache: new Map() };
  return function test(mu) {
    const qObs = muHat > mu ? 0 : Math.max(conditional(model, lik, base, poi, mu).twiceNll - base.twiceNll, 0);
    const qA = Math.max(conditional(model, asimovLik, asimovBase, poi, mu).twiceNll - asimovBase.twiceNll, 0);
    const rootQ = Math.sqrt(qObs);
    const rootA = Math.sqrt(qA);
    const ts = !tilde || rootQ <= rootA || rootA === 0 ? rootQ - rootA : (qObs - qA) / (2 * rootA);
    const clsb = normSf(ts + rootA);
    const clb = normSf(ts);
    return {
      mu,
      cls: clsb / clb,
      clsb,
      clb,
      expected: [2, 1, 0, -1, -2].map((nsig) => normSf(nsig + rootA) / normSf(nsig)),
      q: qObs,
      qAsimov: qA,
      tilde,
    };
  };
}

// first place where y(x) drops below the level: [index of the point below, interpolated x]
function crossing(x, y, level) {
  if (!(y[0] >= level)) return [-1, NaN];
  for (let i = 1; i < x.length; i++) {
    if (y[i] < level) return [i, x[i] + ((level - y[i]) / (y[i - 1] - y[i])) * (x[i - 1] - x[i])];
  }
  return [-1, NaN];
}

// Upper limit on the POI from a scan of CLs. options: level, points, or values for an explicit scan
// (then the limit is interpolated linearly between the scan points, as pyhf does). Without explicit
// values the range is found automatically and each crossing is refined with further tests.
async function limit(model, lik, base, options = {}, ctx = NO_CTX) {
  const poi = model.poiIndex;
  const level = options.level || 0.05;
  await ctx.tick({ label: "limit: background-only fit" });
  const test = tester(model, lik, base);
  let values = options.values;
  if (!values) {
    // find a range in which all curves drop below the level
    const top = model.upper[poi];
    const sigma = base.unc[poi] > 0 ? base.unc[poi] : top / 20;
    let hi = Math.min(top, Math.max(base.pars[poi], 0) + 3 * sigma);
    for (let i = 0; i < 12; i++) {
      await ctx.tick({ label: `limit: looking for the scan range (up to ${hi.toPrecision(3)})` });
      const r = test(hi);
      const worst = Math.max(r.cls, r.expected[4]);
      if (worst > level / 5 && hi < top) hi = Math.min(top, hi * 1.7);
      else if (worst < level / 500) hi *= 0.6;
      else break;
    }
    const n = options.points || 21;
    values = linspace(0, hi, n);
  }
  const results = [];
  for (let i = 0; i < values.length; i++) {
    await ctx.tick({ done: i, total: values.length, label: "limit: scanning the signal strength" });
    results.push(test(values[i]));
  }
  const curves = [(r) => r.cls, 0, 1, 2, 3, 4].map((c) => (typeof c === "number" ? (r) => r.expected[c] : c));
  const limits = [];
  for (let c = 0; c < curves.length; c++) {
    const get = curves[c];
    let [i, x] = crossing(results.map((r) => r.mu), results.map(get), level);
    if (!options.values && i > 0) {
      // regula falsi on ln(CLs) between the two scan points around the crossing
      await ctx.tick({ done: c, total: curves.length, label: "limit: refining the crossings" });
      let a = results[i - 1];
      let b = results[i];
      for (let it = 0; it < 8; it++) {
        const fa = Math.log(get(a) / level);
        const fb = Math.log(Math.max(get(b), 1e-300) / level);
        x = a.mu - (fa * (b.mu - a.mu)) / (fb - fa);
        if (Math.abs(b.mu - a.mu) < 1e-4 * x) break;
        const r = test(x);
        results.push(r);
        if (Math.abs(get(r) / level - 1) < 1e-4) break;
        if (get(r) >= level) a = r;
        else b = r;
      }
    }
    limits.push(x);
  }
  results.sort((p, q) => p.mu - q.mu);
  return {
    level,
    values: results.map((r) => r.mu),
    results,
    observed: limits[0],
    expected: limits.slice(1),
    tilde: results[0].tilde,
  };
}

// Discovery test statistic q0 with observed and expected significance.
async function significance(model, lik, base, ctx = NO_CTX) {
  const poi = model.poiIndex;
  await ctx.tick({ label: "significance: background-only fit" });
  const bkgOnly = conditional(model, lik, base, poi, 0);
  const q0 = base.pars[poi] > 0 ? Math.max(bkgOnly.twiceNll - base.twiceNll, 0) : 0;
  await ctx.tick({ label: "significance: expected" });
  const asimovPars = conditional(model, lik, base, poi, 1).pars;
  const asimovLik = model.likelihood(model.expectedData(asimovPars));
  const asimovBase = { ...base, pars: asimovPars, twiceNll: asimovLik.value(asimovPars), cache: new Map() };
  const q0A = Math.max(conditional(model, asimovLik, asimovBase, poi, 0).twiceNll - asimovBase.twiceNll, 0);
  return {
    q0,
    z: Math.sqrt(q0),
    p0: normSf(Math.sqrt(q0)),
    zExpected: Math.sqrt(q0A),
    p0Expected: normSf(Math.sqrt(q0A)),
  };
}

// Yields per channel, sample and bin with uncertainties from linear error propagation.
// Without a fit result the pre-fit uncertainties of the constraint terms are used. The uncertainties are NaN if
// the fit has no covariance matrix.
function yields(model, pars, base) {
  const N = model.nPars;
  const grad = new Float64Array(N);
  const k = base ? base.inner.length : 0;
  const variance = () => {
    let v = 0;
    if (!base) {
      for (let p = 0; p < N; p++) if (!model.fixed[p] && grad[p] !== 0) v += (grad[p] * model.prefitUnc[p]) ** 2;
      return v;
    }
    if (!base.cov) return NaN;
    const nz = [];
    for (let a = 0; a < k; a++) if (grad[base.inner[a]] !== 0) nz.push(a);
    for (const a of nz) {
      const ga = grad[base.inner[a]];
      const r = a * k;
      let t = 0;
      for (const b of nz) t += base.cov[r + b] * grad[base.inner[b]];
      v += ga * t;
    }
    return Math.max(v, 0);
  };
  model.evaluate(pars);
  const cellNu = Float64Array.from(model.cellNu);
  const w = model.cellWeight;
  const uncertainty = (blocks, lo, hi) => {
    // uncertainty of the sum over the given blocks, restricted to bins lo..hi of the channel
    w.fill(0);
    for (const blk of blocks) for (let b = lo; b < hi; b++) w[blk.cell + b] = 1;
    grad.fill(0);
    model.accumulate(pars, grad, blocks);
    return Math.sqrt(variance());
  };
  const channels = model.channels.map((ch, ci) => {
    const blocks = model.channelBlocks[ci];
    const total = new Array(ch.nBins).fill(0);
    const samples = blocks.map((blk) => {
      const y = Array.from(cellNu.subarray(blk.cell, blk.cell + ch.nBins));
      y.forEach((v, b) => (total[b] += v));
      return { name: model.samples[blk.sample], yields: y, sum: y.reduce((s, v) => s + v, 0), sumUnc: uncertainty([blk], 0, ch.nBins) };
    });
    const data = Array.from(model.observed.subarray(ch.offset, ch.offset + ch.nBins));
    return {
      name: ch.name,
      data,
      dataSum: data.reduce((s, v) => s + v, 0),
      total,
      totalUnc: total.map((_, b) => uncertainty(blocks, b, b + 1)),
      samples,
      sum: total.reduce((s, v) => s + v, 0),
      sumUnc: uncertainty(blocks, 0, ch.nBins),
    };
  });
  w.fill(0);
  return { channels };
}

function correlation(base) {
  const k = base.inner.length;
  const out = new Float64Array(k * k);
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) out[a * k + b] = base.cov[a * k + b] / Math.sqrt(base.cov[a * k + a] * base.cov[b * k + b]);
  }
  return out;
}

export { fit, conditional, scan, scan2d, LEVELS, approximateImpacts, ranking, tester, limit, significance, yields, correlation, covAt };
