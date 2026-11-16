// HistFactory model built from a pyhf workspace: expected yields, likelihood and its gradient.
// Channel, sample and parameter ordering follow pyhf, so vectors compare element by element.
import { lgamma } from "./numerics.js";
import { MODIFIER_TYPES } from "./workspace.js";
const LN2PI = Math.log(2 * Math.PI);

const A_INVERSE = [
  [15 / 16, -15 / 16, -7 / 16, -7 / 16, 1 / 16, -1 / 16],
  [3 / 2, 3 / 2, -9 / 16, 9 / 16, 1 / 16, 1 / 16],
  [-5 / 8, 5 / 8, 5 / 8, 5 / 8, -1 / 8, 1 / 8],
  [-3 / 2, -3 / 2, 7 / 8, -7 / 8, -1 / 8, -1 / 8],
  [3 / 16, -3 / 16, -3 / 16, -3 / 16, 1 / 16, -1 / 16],
  [1 / 2, 1 / 2, -5 / 16, 5 / 16, 1 / 16, 1 / 16],
];
// polynomial coefficients of interpolation code 4 for |alpha| < 1
function code4Coefficients(hi, lo) {
  const lu = Math.log(hi);
  const ld = Math.log(lo);
  const b = [hi - 1, lo - 1, lu * hi, -ld * lo, lu * lu * hi, ld * ld * lo];
  return A_INVERSE.map((row) => row.reduce((s, a, j) => s + a * b[j], 0));
}

const sameList = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function Model(ws, measurementIndex = 0, options = {}) {
  const measurement = ws.measurements[measurementIndex];
  this.measurement = measurement.name;

  // ----- channels, samples, modifiers (all sorted, as in pyhf) -----
  const channelSpecs = {};
  for (const c of ws.channels) channelSpecs[c.name] = c;
  const channelNames = Object.keys(channelSpecs).sort();
  const sampleSet = new Set();
  const modifierKeys = new Map(); // "name\u0000type" -> [name, type]
  for (const c of ws.channels) {
    for (const s of c.samples) {
      sampleSet.add(s.name);
      for (const m of s.modifiers) modifierKeys.set(`${m.name}\u0000${m.type}`, [m.name, m.type]);
    }
  }
  this.samples = [...sampleSet].sort();
  const modifiers = [...modifierKeys.keys()].sort().map((k) => modifierKeys.get(k));

  this.channels = [];
  let offset = 0;
  for (const name of channelNames) {
    const nBins = channelSpecs[name].samples[0].data.length;
    this.channels.push({ name, nBins, offset });
    offset += nBins;
  }
  this.nBins = offset;
  const B = this.nBins;

  // defined (channel, sample) combinations with their modifiers by type
  const defined = this.channels.map((ch) => {
    const bySample = {};
    for (const s of channelSpecs[ch.name].samples) {
      const mods = {};
      for (const m of s.modifiers) mods[`${m.type}/${m.name}`] = m;
      bySample[s.name] = { sample: s, mods };
    }
    return this.samples.map((sn) => bySample[sn] || null);
  });

  // ----- parameter sets -----
  const requirements = new Map(); // name -> list of requirements
  const require = (name, req) => {
    if (!requirements.has(name)) requirements.set(name, []);
    requirements.get(name).push(req);
  };
  const scalar = (kind, init, bounds, extra = {}) => ({ kind, n: 1, init: [init], bounds: [bounds], fixed: [false], ...extra });
  const stat = {}; // staterror name -> {bins: global bin indices, sigmas}
  const shapeBlocks = {}; // shapesys / shapefactor name -> channel index

  for (const type of MODIFIER_TYPES) {
    if (type === "staterror") {
      for (const [name, t] of modifiers) {
        if (t !== type) continue;
        const key = `${type}/${name}`;
        const nomSum = new Float64Array(B);
        const unc2 = new Float64Array(B);
        const mask = new Uint8Array(B);
        this.samples.forEach((sn, si) => {
          if (!this.channels.some((ch, ci) => defined[ci][si] && defined[ci][si].mods[key])) return;
          this.channels.forEach((ch, ci) => {
            const d = defined[ci][si];
            if (!d) return;
            const mod = d.mods[key];
            for (let b = 0; b < ch.nBins; b++) {
              nomSum[ch.offset + b] += d.sample.data[b];
              if (mod) {
                unc2[ch.offset + b] += mod.data[b] * mod.data[b];
                mask[ch.offset + b] = 1;
              }
            }
          });
        });
        const bins = [];
        const sigmas = [];
        for (let b = 0; b < B; b++) {
          if (!mask[b]) continue;
          bins.push(b);
          sigmas.push(nomSum[b] > 0 ? Math.sqrt(unc2[b]) / nomSum[b] : 0);
        }
        const fixed = sigmas.map((s) => s === 0);
        stat[name] = { bins };
        require(name, {
          kind: "normal",
          n: bins.length,
          init: bins.map(() => 1),
          bounds: bins.map(() => [1e-10, 10]),
          fixed,
          aux: bins.map(() => 1),
          sigmas: sigmas.map((s) => (s === 0 ? 1 : s)),
        });
      }
      continue;
    }
    const seen = new Set();
    this.channels.forEach((ch, ci) => {
      this.samples.forEach((sn, si) => {
        const d = defined[ci][si];
        if (!d) return;
        for (const [name, t] of modifiers) {
          if (t !== type) continue;
          const mod = d.mods[`${type}/${name}`];
          if (!mod) continue;
          if (type === "shapesys" || type === "shapefactor") {
            if (name in shapeBlocks && (type === "shapesys" || shapeBlocks[name] !== ci)) {
              throw new Error(
                type === "shapesys"
                  ? `The shapesys modifier "${name}" is used more than once; shapesys modifiers cannot be shared between samples or channels.`
                  : `The shapefactor modifier "${name}" is used in more than one channel, which is not supported.`
              );
            }
            shapeBlocks[name] = ci;
          }
          if (seen.has(name)) continue;
          seen.add(name);
          if (type === "histosys" || type === "normsys") require(name, scalar("normal", 0, [-5, 5], { aux: [0] }));
          else if (type === "normfactor") require(name, scalar("free", 1, [0, 10]));
          else if (type === "lumi") require(name, scalar("normal", null, null, { aux: null, sigmas: null }));
          else if (type === "shapefactor") {
            const bins = d.sample.data;
            require(name, { kind: "free", n: bins.length, init: bins.map(() => 1), bounds: bins.map(() => [0, 10]), fixed: bins.map(() => false) });
          } else if (type === "shapesys") {
            const valid = d.sample.data.map((nom, b) => nom > 0 && mod.data[b] > 0);
            const factors = d.sample.data.map((nom, b) => (valid[b] ? (nom * nom) / (mod.data[b] * mod.data[b]) : 1));
            require(name, {
              kind: "poisson",
              n: factors.length,
              init: factors.map(() => 1),
              bounds: factors.map(() => [1e-10, 10]),
              fixed: valid.map((v) => !v),
              aux: factors,
              factors,
            });
          }
        }
      });
    });
  }
  if (requirements.size === 0) throw new Error("The model has no parameters.");

  const userConfig = {};
  for (const p of measurement.config.parameters) {
    if (p.name in userConfig) throw new Error(`Measurement "${measurement.name}" configures parameter "${p.name}" more than once.`);
    userConfig[p.name] = p;
  }

  this.sets = [];
  const setByName = {};
  let nPars = 0;
  for (const [name, reqs] of requirements) {
    const set = { ...reqs[0] };
    for (const other of reqs.slice(1)) {
      for (const k of ["kind", "n", "init", "bounds", "fixed", "aux", "sigmas", "factors"]) {
        if (!sameList(set[k], other[k])) {
          throw new Error(`The name "${name}" is used by modifiers of incompatible types; use unique modifier names.`);
        }
      }
    }
    const user = userConfig[name] || {};
    const take = (key, userKey, wrap) => {
      if (user[userKey] === undefined) return;
      let v = user[userKey];
      if (wrap) v = v.map((x) => x);
      if (set[key] === undefined) throw new Error(`Parameter "${name}" does not use the "${userKey}" setting.`);
      if (!Array.isArray(v) || v.length !== set.n) {
        throw new Error(`Parameter "${name}": "${userKey}" needs ${set.n} value${set.n === 1 ? "" : "s"}.`);
      }
      set[key] = v;
    };
    take("init", "inits");
    take("bounds", "bounds");
    take("aux", "auxdata");
    take("sigmas", "sigmas");
    take("factors", "factors");
    if (typeof user.fixed === "boolean") set.fixed = new Array(set.n).fill(user.fixed);
    if (set.init === null || set.bounds === null || set.aux === null || set.sigmas === null) {
      throw new Error(`The luminosity parameter "${name}" needs inits, bounds, auxdata and sigmas in the measurement's parameter configuration.`);
    }
    set.name = name;
    set.start = nPars;
    set.types = [];
    nPars += set.n;
    this.sets.push(set);
    setByName[name] = set;
  }
  for (const [name, type] of modifiers) setByName[name].types.push(type);
  this.nPars = nPars;

  // flat per-parameter arrays
  this.names = [];
  this.setIndex = new Int32Array(nPars);
  this.init = new Float64Array(nPars);
  this.lower = new Float64Array(nPars);
  this.upper = new Float64Array(nPars);
  this.fixed = new Uint8Array(nPars);
  this.prefitUnc = new Float64Array(nPars);
  this.auxIndex = new Int32Array(nPars).fill(-1);
  const gaussIdx = [];
  const gaussSigma = [];
  const poisIdx = [];
  const poisTau = [];
  this.auxdata = [];
  this.sets.forEach((set, k) => {
    for (let i = 0; i < set.n; i++) {
      const p = set.start + i;
      this.names.push(set.n === 1 && set.types.every((t) => t !== "staterror" && t !== "shapesys" && t !== "shapefactor") ? set.name : `${set.name}[${i}]`);
      this.setIndex[p] = k;
      this.init[p] = set.init[i];
      this.lower[p] = set.bounds[i][0];
      this.upper[p] = set.bounds[i][1];
      this.fixed[p] = set.fixed[i] ? 1 : 0;
      if (set.kind === "free") continue;
      this.auxIndex[p] = this.auxdata.length;
      this.auxdata.push(set.aux[i]);
      if (set.kind === "normal") {
        gaussIdx.push(p);
        gaussSigma.push(set.sigmas ? set.sigmas[i] : 1);
        this.prefitUnc[p] = set.sigmas ? set.sigmas[i] : 1;
      } else {
        poisIdx.push(p);
        poisTau.push(set.factors[i]);
        this.prefitUnc[p] = 1 / Math.sqrt(set.factors[i]);
      }
    }
  });
  this.gaussIdx = Int32Array.from(gaussIdx);
  this.gaussSigma = Float64Array.from(gaussSigma);
  this.poisIdx = Int32Array.from(poisIdx);
  this.poisTau = Float64Array.from(poisTau);
  this.nAux = this.auxdata.length;

  // parameter of interest
  const poi = options.poi !== undefined ? options.poi : measurement.config.poi;
  this.poiIndex = -1;
  this.poiName = "";
  if (poi) {
    const set = setByName[poi];
    if (!set) throw new Error(`The parameter of interest "${poi}" does not appear in the model.`);
    if (set.n !== 1) throw new Error(`The parameter of interest "${poi}" has several components, which is not supported.`);
    this.poiIndex = set.start;
    this.poiName = poi;
    if (options.poiBounds) {
      this.lower[set.start] = options.poiBounds[0];
      this.upper[set.start] = options.poiBounds[1];
      this.init[set.start] = Math.min(Math.max(this.init[set.start], this.lower[set.start]), this.upper[set.start]);
    }
  }

  // observed data in model order, followed by the auxiliary data
  this.observed = new Float64Array(B + this.nAux);
  this.channels.forEach((ch) => this.observed.set(ws.observations[ch.name], ch.offset));
  this.observed.set(this.auxdata, B);

  // ----- compiled blocks, one per defined (channel, sample) -----
  const statIndex = {};
  for (const name of Object.keys(stat)) {
    const map = new Int32Array(B).fill(-1);
    stat[name].bins.forEach((b, i) => (map[b] = setByName[name].start + i));
    statIndex[name] = map;
  }
  this.blocks = [];
  this.channelBlocks = this.channels.map(() => []);
  let cell = 0;
  this.channels.forEach((ch, ci) => {
    this.samples.forEach((sn, si) => {
      const d = defined[ci][si];
      if (!d) return;
      const n = ch.nBins;
      const nominal = Float64Array.from(d.sample.data);
      const scal = [];
      const ns = [];
      const hs = [];
      const gam = [];
      for (const [name, type] of modifiers) {
        const mod = d.mods[`${type}/${name}`];
        if (!mod) continue;
        const start = setByName[name].start;
        if (type === "normfactor" || type === "lumi") scal.push(start);
        else if (type === "normsys") ns.push({ p: start, hi: mod.data.hi, lo: mod.data.lo });
        else if (type === "histosys") hs.push({ p: start, hi: mod.data.hi_data, lo: mod.data.lo_data });
        else if (type === "staterror") gam.push(statIndex[name].slice(ch.offset, ch.offset + n));
        else gam.push(Int32Array.from({ length: n }, (_, b) => start + b));
      }
      const block = {
        id: this.blocks.length,
        channel: ci,
        sample: si,
        offset: ch.offset,
        n,
        cell,
        nominal,
        scal: Int32Array.from(scal),
        nsPar: Int32Array.from(ns.map((m) => m.p)),
        nsLnHi: Float64Array.from(ns.map((m) => Math.log(m.hi))),
        nsLnLo: Float64Array.from(ns.map((m) => Math.log(m.lo))),
        nsCoef: Float64Array.from(ns.flatMap((m) => code4Coefficients(m.hi, m.lo))),
        nsVal: new Float64Array(ns.length),
        nsDer: new Float64Array(ns.length),
        hsPar: Int32Array.from(hs.map((m) => m.p)),
        hsUp: Float64Array.from(hs.flatMap((m) => m.hi.map((v, b) => v - nominal[b]))),
        hsDn: Float64Array.from(hs.flatMap((m) => m.lo.map((v, b) => nominal[b] - v))),
        gam,
      };
      this.blocks.push(block);
      this.channelBlocks[ci].push(block);
      cell += n;
    });
  });
  this.nCells = cell;

  // scratch space filled by evaluate()
  this.nu = new Float64Array(B); // expected events per bin
  this.cellNu = new Float64Array(cell); // per sample and bin
  this.cellBase = new Float64Array(cell); // nominal plus additive shifts
  this.cellGamma = new Float64Array(cell); // product of per-bin factors
  this.blockFactor = new Float64Array(this.blocks.length); // product of overall factors
  this.cellWeight = new Float64Array(cell);
}

// expected events for the given parameters; results are left in this.nu and this.cellNu
Model.prototype.evaluate = function (pars) {
  const { nu, cellNu, cellBase, cellGamma, blockFactor } = this;
  nu.fill(0);
  for (const blk of this.blocks) {
    let F = 1;
    for (let k = 0; k < blk.scal.length; k++) F *= pars[blk.scal[k]];
    for (let k = 0; k < blk.nsPar.length; k++) {
      const a = pars[blk.nsPar[k]];
      let v;
      let d;
      if (a >= 1) {
        v = Math.exp(a * blk.nsLnHi[k]);
        d = blk.nsLnHi[k] * v;
      } else if (a <= -1) {
        v = Math.exp(-a * blk.nsLnLo[k]);
        d = -blk.nsLnLo[k] * v;
      } else {
        const c = blk.nsCoef;
        const o = 6 * k;
        v = 1 + a * (c[o] + a * (c[o + 1] + a * (c[o + 2] + a * (c[o + 3] + a * (c[o + 4] + a * c[o + 5])))));
        d = c[o] + a * (2 * c[o + 1] + a * (3 * c[o + 2] + a * (4 * c[o + 3] + a * (5 * c[o + 4] + a * 6 * c[o + 5]))));
      }
      blk.nsVal[k] = v;
      blk.nsDer[k] = d;
      F *= v;
    }
    blockFactor[blk.id] = F;
    const n = blk.n;
    const c0 = blk.cell;
    for (let b = 0; b < n; b++) cellBase[c0 + b] = blk.nominal[b];
    for (let k = 0; k < blk.hsPar.length; k++) {
      const a = pars[blk.hsPar[k]];
      const o = k * n;
      if (a > 1) {
        for (let b = 0; b < n; b++) cellBase[c0 + b] += blk.hsUp[o + b] * a;
      } else if (a < -1) {
        for (let b = 0; b < n; b++) cellBase[c0 + b] += blk.hsDn[o + b] * a;
      } else {
        const poly = a * a * (15 + a * a * (-10 + a * a * 3));
        for (let b = 0; b < n; b++) {
          const up = blk.hsUp[o + b];
          const dn = blk.hsDn[o + b];
          cellBase[c0 + b] += 0.5 * (up + dn) * a + 0.0625 * (up - dn) * poly;
        }
      }
    }
    for (let b = 0; b < n; b++) {
      let g = 1;
      for (let k = 0; k < blk.gam.length; k++) {
        const idx = blk.gam[k][b];
        if (idx >= 0) g *= pars[idx];
      }
      cellGamma[c0 + b] = g;
      const v = cellBase[c0 + b] * F * g;
      cellNu[c0 + b] = v;
      nu[blk.offset + b] += v;
    }
  }
  return nu;
};

// adds to grad the derivative of sum_cells cellWeight * cellNu with respect to the parameters;
// evaluate(pars) must have been called before with the same parameters
Model.prototype.accumulate = function (pars, grad, blocks) {
  const { cellNu, cellBase, cellGamma, blockFactor, cellWeight } = this;
  for (const blk of blocks || this.blocks) {
    const n = blk.n;
    const c0 = blk.cell;
    const F = blockFactor[blk.id];
    let T = 0;
    let U = 0;
    let any = false;
    for (let b = 0; b < n; b++) {
      const w = cellWeight[c0 + b];
      if (w === 0) continue;
      any = true;
      T += w * cellNu[c0 + b];
      U += w * cellBase[c0 + b] * cellGamma[c0 + b];
    }
    if (!any) continue;
    for (let k = 0; k < blk.scal.length; k++) {
      const p = blk.scal[k];
      if (pars[p] !== 0) grad[p] += T / pars[p];
      else {
        let rest = 1;
        for (let j = 0; j < blk.scal.length; j++) if (j !== k) rest *= pars[blk.scal[j]];
        for (let j = 0; j < blk.nsPar.length; j++) rest *= blk.nsVal[j];
        grad[p] += U * rest;
      }
    }
    for (let k = 0; k < blk.nsPar.length; k++) grad[blk.nsPar[k]] += (T * blk.nsDer[k]) / blk.nsVal[k];
    for (let k = 0; k < blk.hsPar.length; k++) {
      const a = pars[blk.hsPar[k]];
      const o = k * n;
      let s = 0;
      if (a > 1) {
        for (let b = 0; b < n; b++) s += cellWeight[c0 + b] * cellGamma[c0 + b] * blk.hsUp[o + b];
      } else if (a < -1) {
        for (let b = 0; b < n; b++) s += cellWeight[c0 + b] * cellGamma[c0 + b] * blk.hsDn[o + b];
      } else {
        const dpoly = a * (30 + a * a * (-40 + a * a * 18));
        for (let b = 0; b < n; b++) {
          const up = blk.hsUp[o + b];
          const dn = blk.hsDn[o + b];
          s += cellWeight[c0 + b] * cellGamma[c0 + b] * (0.5 * (up + dn) + 0.0625 * (up - dn) * dpoly);
        }
      }
      grad[blk.hsPar[k]] += F * s;
    }
    for (let k = 0; k < blk.gam.length; k++) {
      const list = blk.gam[k];
      for (let b = 0; b < n; b++) {
        const idx = list[b];
        const w = cellWeight[c0 + b];
        if (idx < 0 || w === 0) continue;
        if (pars[idx] !== 0) grad[idx] += (w * cellNu[c0 + b]) / pars[idx];
        else {
          let rest = 1;
          for (let j = 0; j < blk.gam.length; j++) if (j !== k && blk.gam[j][b] >= 0) rest *= pars[blk.gam[j][b]];
          grad[idx] += w * cellBase[c0 + b] * F * rest;
        }
      }
    }
  }
};

// expected main and auxiliary data, as pyhf's Model.expected_data
Model.prototype.expectedData = function (pars) {
  const out = new Float64Array(this.nBins + this.nAux);
  out.set(this.evaluate(pars));
  for (let k = 0; k < this.gaussIdx.length; k++) out[this.nBins + this.auxIndex[this.gaussIdx[k]]] = pars[this.gaussIdx[k]];
  for (let k = 0; k < this.poisIdx.length; k++) out[this.nBins + this.auxIndex[this.poisIdx[k]]] = pars[this.poisIdx[k]] * this.poisTau[k];
  return out;
};

// -2 ln L for a data vector (main data followed by auxiliary data)
Model.prototype.likelihood = function (data) {
  const m = this;
  const B = m.nBins;
  let constant = 0;
  for (let b = 0; b < B; b++) constant += 2 * lgamma(data[b] + 1);
  for (let k = 0; k < m.gaussIdx.length; k++) constant += 2 * Math.log(m.gaussSigma[k]) + LN2PI;
  for (let k = 0; k < m.poisIdx.length; k++) constant += 2 * lgamma(data[B + m.auxIndex[m.poisIdx[k]]] + 1);

  function value(pars, grad) {
    const nu = m.evaluate(pars);
    let t = constant;
    for (let b = 0; b < B; b++) {
      const v = nu[b];
      if (v > 0) t += 2 * (v - data[b] * Math.log(v));
      else if (!(v === 0 && data[b] === 0)) return Infinity;
    }
    for (let k = 0; k < m.gaussIdx.length; k++) {
      const p = m.gaussIdx[k];
      const z = (data[B + m.auxIndex[p]] - pars[p]) / m.gaussSigma[k];
      t += z * z;
    }
    for (let k = 0; k < m.poisIdx.length; k++) {
      const p = m.poisIdx[k];
      const a = data[B + m.auxIndex[p]];
      const rate = pars[p] * m.poisTau[k];
      if (rate > 0) t += 2 * (rate - a * Math.log(rate));
      else if (!(rate === 0 && a === 0)) return Infinity;
    }
    if (!Number.isFinite(t)) return Infinity;
    if (!grad) return t;

    grad.fill(0);
    for (const blk of m.blocks) {
      for (let b = 0; b < blk.n; b++) {
        const v = nu[blk.offset + b];
        m.cellWeight[blk.cell + b] = v > 0 ? 2 * (1 - data[blk.offset + b] / v) : 2;
      }
    }
    m.accumulate(pars, grad);
    for (let k = 0; k < m.gaussIdx.length; k++) {
      const p = m.gaussIdx[k];
      grad[p] += (2 * (pars[p] - data[B + m.auxIndex[p]])) / (m.gaussSigma[k] * m.gaussSigma[k]);
    }
    for (let k = 0; k < m.poisIdx.length; k++) {
      const p = m.poisIdx[k];
      const a = data[B + m.auxIndex[p]];
      grad[p] += 2 * (m.poisTau[k] - (pars[p] > 0 ? a / pars[p] : 0));
    }
    return t;
  }
  return { value, data };
};

export { Model };
