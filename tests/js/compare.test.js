// Compares @histopresto/core with reference numbers from pyhf (tests/reference/reference.json).
// Run with: pixi run test-js
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import * as HF from "@histopresto/core";

const REF = path.join(import.meta.dirname, "../reference");
const reference = JSON.parse(fs.readFileSync(path.join(REF, "reference.json")));

let passed = 0;
const failures = [];
function close(label, got, want, rel, abs = 0) {
  const ok = Number.isFinite(got) && Math.abs(got - want) <= abs + rel * Math.abs(want);
  if (ok) passed++;
  else failures.push(`${label}: got ${got}, expected ${want}`);
}
function same(label, got, want) {
  if (JSON.stringify(got) === JSON.stringify(want)) passed++;
  else failures.push(`${label}: got ${JSON.stringify(got).slice(0, 200)}, expected ${JSON.stringify(want).slice(0, 200)}`);
}

test("agreement with pyhf", async () => {
  for (const [name, ref] of Object.entries(reference)) {
    const t0 = Date.now();
    const ws = HF.parseWorkspace(fs.readFileSync(path.join(REF, "workspaces", name + ".json"), "utf8"));
    const model = new HF.Model(ws, 0);
    const N = model.nPars;

    // structure
    same(`${name} parameter names`, model.names, ref.par_names);
    same(`${name} channels`, model.channels.map((c) => c.name), ref.channels);
    same(`${name} samples`, model.samples, ref.samples);
    same(`${name} fixed`, Array.from(model.fixed, Boolean), ref.fixed);
    close(`${name} poi index`, model.poiIndex, ref.poi_index, 0);
    ref.init.forEach((v, i) => close(`${name} init ${i}`, model.init[i], v, 1e-12));
    ref.bounds.forEach((b, i) => {
      close(`${name} lower ${i}`, model.lower[i], b[0], 1e-12);
      close(`${name} upper ${i}`, model.upper[i], b[1], 1e-12);
    });
    ref.auxdata.forEach((v, i) => close(`${name} auxdata ${i}`, model.auxdata[i], v, 1e-12));
    ref.data.forEach((v, i) => close(`${name} data ${i}`, model.observed[i], v, 1e-12));

    // yields, likelihood and gradient at random points
    const lik = model.likelihood(model.observed);
    ref.points.forEach((pt, k) => {
      const exp = model.expectedData(pt.pars);
      pt.expected_data.forEach((v, i) => close(`${name} point ${k} expected ${i}`, exp[i], v, 1e-10, 1e-12));
      const grad = new Float64Array(N);
      const f = lik.value(pt.pars, grad);
      close(`${name} point ${k} logpdf`, -0.5 * f, pt.logpdf, 1e-10, 1e-9);
      for (let i = 0; i < N; i += Math.max(1, Math.floor(N / 40))) {
        const h = 1e-4; // large enough that rounding in the likelihood value does not dominate
        const up = pt.pars.slice();
        const dn = pt.pars.slice();
        up[i] += h;
        dn[i] -= h;
        const numeric = (lik.value(up) - lik.value(dn)) / (2 * h);
        close(`${name} point ${k} gradient ${model.names[i]}`, grad[i], numeric, 1e-5, 1e-5 + (1e-12 * Math.abs(f)) / h);
      }
    });

    // maximum likelihood fit
    const t1 = Date.now();
    const base = await HF.infer.fit(model, lik);
    const tFit = Date.now() - t1;
    if (!base.status.converged) failures.push(`${name} fit not converged: ${JSON.stringify(base.status)}`);
    if (!ref.fit) {
      // only a minimum from scipy is available: the fit must be at least as good, and not far below
      if (base.twiceNll <= ref.scipy_twice_nll + 1e-3 && base.twiceNll > ref.scipy_twice_nll - 0.5) passed++;
      else failures.push(`${name} fit -2lnL: got ${base.twiceNll}, scipy found ${ref.scipy_twice_nll}`);
      console.log(`${name}: ${N} parameters, ${model.nBins} bins, fit ${tFit} ms, -2lnL ${base.twiceNll.toFixed(4)} (pyhf with scipy: ${ref.scipy_twice_nll.toFixed(4)})`);
      continue;
    }
    close(`${name} fit -2lnL`, base.twiceNll, ref.fit.twice_nll, 0, 2e-4);
    for (let i = 0; i < N; i++) {
      const scale = Math.max(ref.fit.uncertainty[i], 1e-3);
      close(`${name} bestfit ${model.names[i]}`, base.pars[i], ref.fit.bestfit[i], 0, 0.01 * scale);
      // Minuit's uncertainties: only where the parameter range is wide compared to the uncertainty
      const wide = ref.fit.uncertainty[i] < 0.1 * (model.upper[i] - model.lower[i]);
      if (!model.fixed[i] && !base.atBound[i] && wide) close(`${name} uncertainty (Minuit) ${model.names[i]}`, base.unc[i], ref.fit.uncertainty[i], 0.03, 1e-6);
    }
    if (ref.hesse) {
      same(`${name} parameters with a covariance`, Array.from(base.inner), ref.hesse.free);
      const corr = HF.infer.correlation(base);
      const k = base.inner.length;
      for (let a = 0; a < k; a++) {
        close(`${name} uncertainty ${model.names[base.inner[a]]}`, base.unc[base.inner[a]], ref.hesse.uncertainty[a], 2e-3);
        for (let b = 0; b < a; b++) close(`${name} correlation ${a},${b}`, corr[a * k + b], ref.hesse.correlation[a][b], 0, 2e-3);
      }
    }

    for (const r of ref.fixed_poi) {
      const c = HF.infer.conditional(model, lik, base, model.poiIndex, r.mu);
      close(`${name} fixed POI ${r.mu} -2lnL`, c.twiceNll, r.twice_nll, 0, 1e-3);
    }

    if (ref.hypotest) {
      const test = HF.infer.tester(model, lik, base);
      for (const r of ref.hypotest) {
        const t = test(r.mu);
        close(`${name} qmu_tilde(${r.mu})`, t.q, r.qmu_tilde, 2e-3, 1e-3);
        close(`${name} CLs(${r.mu})`, t.cls, r.cls, 5e-3, 1e-7);
        close(`${name} CLs+b(${r.mu})`, t.clsb, r.clsb, 5e-3, 1e-7);
        close(`${name} CLb(${r.mu})`, t.clb, r.clb, 5e-3, 1e-7);
        r.expected.forEach((v, i) => close(`${name} expected CLs(${r.mu}) ${i}`, t.expected[i], v, 5e-3, 1e-7));
      }
      const lim = await HF.infer.limit(model, lik, base, { values: ref.limit.scan });
      close(`${name} observed limit`, lim.observed, ref.limit.observed, 2e-3);
      ref.limit.expected.forEach((v, i) => close(`${name} expected limit ${i}`, lim.expected[i], v, 2e-3));
      if (ref.limit_fine) {
        const auto = await HF.infer.limit(model, lik, base);
        close(`${name} observed limit (automatic range)`, auto.observed, ref.limit_fine.observed, 2e-3);
        ref.limit_fine.expected.forEach((v, i) => close(`${name} expected limit ${i} (automatic range)`, auto.expected[i], v, 2e-3));
      }

      const sig = await HF.infer.significance(model, lik, base);
      close(`${name} q0`, sig.q0, ref.discovery.q0, 2e-3, 1e-3);
      close(`${name} p0`, sig.p0, ref.discovery.p0, 5e-3, 1e-9);
      close(`${name} expected p0`, sig.p0Expected, ref.discovery.p0_expected, 5e-3, 1e-9);

      // same model with negative signal strengths allowed
      if (ref.hypotest_q.length) {
        const wide = new HF.Model(ws, 0, { poiBounds: [-5, model.upper[model.poiIndex]] });
        const wlik = wide.likelihood(wide.observed);
        const wbase = await HF.infer.fit(wide, wlik);
        const wtest = HF.infer.tester(wide, wlik, wbase);
        for (const r of ref.hypotest_q) {
          const t = wtest(r.mu);
          close(`${name} q: CLs(${r.mu})`, t.cls, r.cls, 5e-3, 1e-7);
          r.expected.forEach((v, i) => close(`${name} q: expected CLs(${r.mu}) ${i}`, t.expected[i], v, 5e-3, 1e-7));
        }
      }
    }
    console.log(`${name}: ${N} parameters, ${model.nBins} bins, fit ${tFit} ms (${base.status.iterations} iterations), all checks ${Date.now() - t0} ms`);
  }
  // measurements: no parameter of interest, a parameter held constant, and inputs that must be refused
  {
    const raw = JSON.parse(fs.readFileSync(path.join(REF, "workspaces", "everything.json"), "utf8"));
    const ref = reference.everything;
    const pars = raw.measurements[0].config.parameters;
    raw.measurements.push({ name: "no poi", config: { poi: "", parameters: pars } });
    raw.measurements.push({ name: "mu fixed", config: { poi: "mu", parameters: pars.filter((q) => q.name !== "mu").concat([{ name: "mu", inits: [2.0], fixed: true }]) } });
    const ws = HF.parseWorkspace(raw);
    const noPoi = new HF.Model(ws, 1);
    close("no POI: index", noPoi.poiIndex, -1, 0);
    close("no POI: fit -2lnL", (await HF.infer.fit(noPoi, noPoi.likelihood(noPoi.observed))).twiceNll, ref.fit.twice_nll, 0, 2e-4);
    const frozen = new HF.Model(ws, 2);
    const fit = await HF.infer.fit(frozen, frozen.likelihood(frozen.observed));
    close("fixed POI by measurement: -2lnL", fit.twiceNll, ref.fixed_poi.find((r) => r.mu === 2).twice_nll, 0, 1e-3);
    close("fixed POI by measurement: value", fit.pars[frozen.poiIndex], 2, 0);
    const refused = (label, change, base = raw) => {
      const copy = JSON.parse(JSON.stringify(base));
      change(copy);
      try {
        new HF.Model(HF.parseWorkspace(copy), 0);
        failures.push(`${label}: accepted`);
      } catch (e) {
        passed++;
      }
    };
    refused("missing observation", (w) => w.observations.pop());
    refused("wrong number of bins", (w) => w.channels[0].samples[0].data.push(1));
    refused("unknown modifier", (w) => (w.channels[0].samples[0].modifiers[0].type = "other"));
    refused("shapesys shared between samples", (w) => w.channels[0].samples[1].modifiers.push({ name: "fake_bins", type: "shapesys", data: [1, 1, 1] }));
    refused("normfactor and normsys with one name", (w) => w.channels[0].samples[1].modifiers.push({ name: "mu", type: "normsys", data: { hi: 1.1, lo: 0.9 } }));
    refused("lumi without configuration", (w) => (w.measurements[0].config.parameters = []));
    refused("unknown POI", (w) => (w.measurements[0].config.poi = "nothing"));
    // expressions
    const pure = JSON.parse(fs.readFileSync(path.join(REF, "workspaces", "purefunc.json"), "utf8"));
    refused("unknown expression language", (w) => (w.bindings[0].language = "python"), pure);
    refused("more names than expressions", (w) => w.bindings[1].names.push("other"), pure);
    refused("circular bindings", (w) => (w.bindings[0].expression = "(kappa + func3, alpha)"), pure);
    refused("purefunc without binding", (w) => w.bindings.pop(), pure);
    refused("unsupported function", (w) => (w.bindings[1].expression = "gamma(theta)"), pure);
    refused("expression with a syntax error", (w) => (w.bindings[1].expression = "func1 + * theta"), pure);
  }
  console.log(`${passed} checks passed, ${failures.length} failed`);
  for (const f of failures.slice(0, 40)) console.log("  FAIL " + f);
  assert.equal(failures.length, 0);
  assert.equal(passed, 7318);
});
