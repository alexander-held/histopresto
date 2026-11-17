"""Compares histopresto with reference numbers from pyhf in tests/reference/reference.json (same checks as tests/js)."""

import copy
import json
import math
import pathlib
import time

import numpy as np

import histopresto
from histopresto import infer

REF = pathlib.Path(__file__).parents[1] / "reference"


def test_agreement_with_pyhf():
    reference = json.loads((REF / "reference.json").read_text())
    passed = 0
    failures = []

    def close(label, got, want, rel, abs_=0.0):
        nonlocal passed
        if math.isfinite(got) and abs(got - want) <= abs_ + rel * abs(want):
            passed += 1
        else:
            failures.append(f"{label}: got {got}, expected {want}")

    def same(label, got, want):
        nonlocal passed
        if got == want:
            passed += 1
        else:
            failures.append(f"{label}: got {str(got)[:200]}, expected {str(want)[:200]}")

    for name, ref in reference.items():
        t0 = time.perf_counter()
        source = (REF / "workspaces" / f"{name}.json").read_text()
        ws = histopresto.parse_workspace(source)
        model = histopresto.Model(ws, 0)
        N = model.n_pars

        # structure
        same(f"{name} parameter names", model.names, ref["par_names"])
        same(f"{name} channels", [c.name for c in model.channels], ref["channels"])
        same(f"{name} samples", model.samples, ref["samples"])
        same(f"{name} fixed", [bool(f) for f in model.fixed], ref["fixed"])
        close(f"{name} poi index", model.poi_index, ref["poi_index"], 0)
        for i, v in enumerate(ref["init"]):
            close(f"{name} init {i}", model.init[i], v, 1e-12)
        for i, b in enumerate(ref["bounds"]):
            close(f"{name} lower {i}", model.lower[i], b[0], 1e-12)
            close(f"{name} upper {i}", model.upper[i], b[1], 1e-12)
        for i, v in enumerate(ref["auxdata"]):
            close(f"{name} auxdata {i}", model.auxdata[i], v, 1e-12)
        for i, v in enumerate(ref["data"]):
            close(f"{name} data {i}", model.observed[i], v, 1e-12)

        # yields, likelihood and gradient at random points
        lik = model.likelihood(model.observed)
        for k, pt in enumerate(ref["points"]):
            pars = np.array(pt["pars"], dtype=float)
            exp = model.expected_data(pars)
            for i, v in enumerate(pt["expected_data"]):
                close(f"{name} point {k} expected {i}", exp[i], v, 1e-10, 1e-12)
            grad = np.zeros(N)
            f = lik.value(pars, grad)
            close(f"{name} point {k} logpdf", -0.5 * f, pt["logpdf"], 1e-10, 1e-9)
            for i in range(0, N, max(1, N // 40)):
                h = 1e-4  # large enough that rounding in the likelihood value does not dominate
                up = pars.copy()
                dn = pars.copy()
                up[i] += h
                dn[i] -= h
                numeric = (lik.value(up) - lik.value(dn)) / (2 * h)
                close(f"{name} point {k} gradient {model.names[i]}", grad[i], numeric, 1e-5, 1e-5 + (1e-12 * abs(f)) / h)

        # maximum likelihood fit
        t1 = time.perf_counter()
        base = infer.fit(model, lik)
        t_fit = 1e3 * (time.perf_counter() - t1)
        if not base.status.converged:
            failures.append(f"{name} fit not converged: {base.status}")
        if "fit" not in ref:
            # only a minimum from scipy is available: the fit must be at least as good, and not far below
            if ref["scipy_twice_nll"] - 0.5 < base.twice_nll <= ref["scipy_twice_nll"] + 1e-3:
                passed += 1
            else:
                failures.append(f"{name} fit -2lnL: got {base.twice_nll}, scipy found {ref['scipy_twice_nll']}")
            print(f"{name}: {N} parameters, {model.n_bins} bins, fit {t_fit:.1f} ms, -2lnL {base.twice_nll:.4f} (pyhf with scipy: {ref['scipy_twice_nll']:.4f})")
            continue
        close(f"{name} fit -2lnL", base.twice_nll, ref["fit"]["twice_nll"], 0, 2e-4)
        for i in range(N):
            scale = max(ref["fit"]["uncertainty"][i], 1e-3)
            close(f"{name} bestfit {model.names[i]}", base.pars[i], ref["fit"]["bestfit"][i], 0, 0.01 * scale)
            # Minuit's uncertainties: only where the parameter range is wide compared to the uncertainty
            wide = ref["fit"]["uncertainty"][i] < 0.1 * (model.upper[i] - model.lower[i])
            if not model.fixed[i] and not base.at_bound[i] and wide:
                close(f"{name} uncertainty (Minuit) {model.names[i]}", base.unc[i], ref["fit"]["uncertainty"][i], 0.03, 1e-6)
        if ref.get("hesse"):
            same(f"{name} parameters with a covariance", base.inner.tolist(), ref["hesse"]["free"])
            corr = infer.correlation(base)
            for a, p in enumerate(base.inner):
                close(f"{name} uncertainty {model.names[p]}", base.unc[p], ref["hesse"]["uncertainty"][a], 2e-3)
                for b in range(a):
                    close(f"{name} correlation {a},{b}", corr[a, b], ref["hesse"]["correlation"][a][b], 0, 2e-3)

        for r in ref["fixed_poi"]:
            c = infer.conditional(model, lik, base, model.poi_index, r["mu"])
            close(f"{name} fixed POI {r['mu']} -2lnL", c.twice_nll, r["twice_nll"], 0, 1e-3)

        if ref.get("hypotest"):
            test = infer.tester(model, lik, base)
            for r in ref["hypotest"]:
                t = test(r["mu"])
                close(f"{name} qmu_tilde({r['mu']})", t.q, r["qmu_tilde"], 2e-3, 1e-3)
                close(f"{name} CLs({r['mu']})", t.cls, r["cls"], 5e-3, 1e-7)
                close(f"{name} CLs+b({r['mu']})", t.clsb, r["clsb"], 5e-3, 1e-7)
                close(f"{name} CLb({r['mu']})", t.clb, r["clb"], 5e-3, 1e-7)
                for i, v in enumerate(r["expected"]):
                    close(f"{name} expected CLs({r['mu']}) {i}", t.expected[i], v, 5e-3, 1e-7)
            lim = infer.limit(model, lik, base, values=ref["limit"]["scan"])
            close(f"{name} observed limit", lim.observed, ref["limit"]["observed"], 2e-3)
            for i, v in enumerate(ref["limit"]["expected"]):
                close(f"{name} expected limit {i}", lim.expected[i], v, 2e-3)
            if ref.get("limit_fine"):
                auto = infer.limit(model, lik, base)
                close(f"{name} observed limit (automatic range)", auto.observed, ref["limit_fine"]["observed"], 2e-3)
                for i, v in enumerate(ref["limit_fine"]["expected"]):
                    close(f"{name} expected limit {i} (automatic range)", auto.expected[i], v, 2e-3)

            sig = infer.significance(model, lik, base)
            close(f"{name} q0", sig.q0, ref["discovery"]["q0"], 2e-3, 1e-3)
            close(f"{name} p0", sig.p0, ref["discovery"]["p0"], 5e-3, 1e-9)
            close(f"{name} expected p0", sig.p0_expected, ref["discovery"]["p0_expected"], 5e-3, 1e-9)

            # same model with negative signal strengths allowed
            if ref["hypotest_q"]:
                wide = histopresto.Model(ws, 0, poi_bounds=[-5, model.upper[model.poi_index]])
                wlik = wide.likelihood(wide.observed)
                wtest = infer.tester(wide, wlik, infer.fit(wide, wlik))
                for r in ref["hypotest_q"]:
                    t = wtest(r["mu"])
                    close(f"{name} q: CLs({r['mu']})", t.cls, r["cls"], 5e-3, 1e-7)
                    for i, v in enumerate(r["expected"]):
                        close(f"{name} q: expected CLs({r['mu']}) {i}", t.expected[i], v, 5e-3, 1e-7)
        print(f"{name}: {N} parameters, {model.n_bins} bins, fit {t_fit:.1f} ms ({base.status.iterations} iterations), all checks {1e3 * (time.perf_counter() - t0):.0f} ms")

    # measurements: no parameter of interest, a parameter held constant, and inputs that must be refused
    raw = json.loads((REF / "workspaces" / "everything.json").read_text())
    ref = reference["everything"]
    pars = raw["measurements"][0]["config"]["parameters"]
    raw["measurements"].append({"name": "no poi", "config": {"poi": "", "parameters": pars}})
    raw["measurements"].append(
        {"name": "mu fixed", "config": {"poi": "mu", "parameters": [q for q in pars if q["name"] != "mu"] + [{"name": "mu", "inits": [2.0], "fixed": True}]}}
    )
    ws = histopresto.parse_workspace(raw)
    no_poi = histopresto.Model(ws, 1)
    close("no POI: index", no_poi.poi_index, -1, 0)
    close("no POI: fit -2lnL", infer.fit(no_poi, no_poi.likelihood(no_poi.observed)).twice_nll, ref["fit"]["twice_nll"], 0, 2e-4)
    frozen = histopresto.Model(ws, 2)
    fit = infer.fit(frozen, frozen.likelihood(frozen.observed))
    close("fixed POI by measurement: -2lnL", fit.twice_nll, next(r for r in ref["fixed_poi"] if r["mu"] == 2)["twice_nll"], 0, 1e-3)
    close("fixed POI by measurement: value", fit.pars[frozen.poi_index], 2, 0)

    def refused(label, change, base=raw):
        nonlocal passed
        w = copy.deepcopy(base)
        change(w)
        try:
            histopresto.Model(histopresto.parse_workspace(w), 0)
            failures.append(f"{label}: accepted")
        except ValueError:
            passed += 1

    def set_type(w):
        w["channels"][0]["samples"][0]["modifiers"][0]["type"] = "other"

    def clear_parameters(w):
        w["measurements"][0]["config"]["parameters"] = []

    def set_poi(w):
        w["measurements"][0]["config"]["poi"] = "nothing"

    refused("missing observation", lambda w: w["observations"].pop())
    refused("wrong number of bins", lambda w: w["channels"][0]["samples"][0]["data"].append(1))
    refused("unknown modifier", set_type)
    refused("shapesys shared between samples", lambda w: w["channels"][0]["samples"][1]["modifiers"].append({"name": "fake_bins", "type": "shapesys", "data": [1, 1, 1]}))
    refused("normfactor and normsys with one name", lambda w: w["channels"][0]["samples"][1]["modifiers"].append({"name": "mu", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}))
    refused("lumi without configuration", clear_parameters)
    refused("unknown POI", set_poi)

    # expressions
    pure = json.loads((REF / "workspaces" / "purefunc.json").read_text())

    def set_binding(index, key, value):
        return lambda w: w["bindings"][index].update({key: value})

    refused("unknown expression language", set_binding(0, "language", "python"), pure)
    refused("more names than expressions", lambda w: w["bindings"][1]["names"].append("other"), pure)
    refused("circular bindings", set_binding(0, "expression", "(kappa + func3, alpha)"), pure)
    refused("purefunc without binding", lambda w: w["bindings"].pop(), pure)
    refused("unsupported function", set_binding(1, "expression", "gamma(theta)"), pure)
    refused("expression with a syntax error", set_binding(1, "expression", "func1 + * theta"), pure)

    print(f"{passed} checks passed, {len(failures)} failed")
    assert not failures, "\n".join(failures[:40])
    assert passed == 7369
