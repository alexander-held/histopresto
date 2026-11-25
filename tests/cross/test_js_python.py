"""The JavaScript and the Python + C implementation follow the same algorithm step by step: same number of
iterations, results equal up to rounding."""

import json
import math
import pathlib
import subprocess

import pytest

import histopresto
from histopresto import infer

REPO = pathlib.Path(__file__).parents[2]
WORKSPACES = {
    "counting": REPO / "js/app/examples/counting.json",
    "search": REPO / "js/app/examples/search.json",
    "large": REPO / "tests/reference/workspaces/large.json",
    "purefunc": REPO / "tests/reference/workspaces/purefunc.json",
    "purefunc_mixed": REPO / "tests/reference/workspaces/purefunc_mixed.json",
}


def python_results(path):
    model = histopresto.Model(histopresto.parse_workspace(path.read_text()), 0)
    lik = model.likelihood(model.observed)
    base = infer.fit(model, lik)
    test = infer.tester(model, lik, base)(1.0)
    limit = infer.limit(model, lik, base)
    significance = infer.significance(model, lik, base)
    # scans of the two parameters with the smallest index that the fit determined
    def grid(i, n):
        return [min(max(base.pars[i] + ((k - (n - 1) / 2) * 5 * base.unc[i]) / (n - 1), model.lower[i]), model.upper[i]) for k in range(n)]

    p = int(base.inner[0])
    scan = infer.scan(model, lik, base, p, grid(p, 11))
    scan2d = []
    if len(base.inner) > 1:
        q = int(base.inner[1])
        scan2d = [v for row in infer.scan2d(model, lik, base, [p, q], grid(p, 4), grid(q, 4)).twice_delta_nll for v in row]
    return {
        "parameters": list(model.names),
        "iterations": base.status.iterations,
        "twice_nll": base.twice_nll,
        "pars": base.pars.tolist(),
        "unc": base.unc.tolist(),
        "cls": [test.cls, *test.expected],
        "limit": [limit.observed, *limit.expected],
        "limit_tests": len(limit.results),
        "z": [significance.z, significance.z_expected],
        "scan": scan.twice_delta_nll,
        "intervals": [v if math.isfinite(v) else str(v).replace("inf", "Infinity").replace("nan", "NaN") for iv in scan.intervals for r in iv.ranges for v in r],
        "scan_unc": float(base.unc[p]),
        "scan2d": scan2d,
    }


def worst(a, b, scale=None):
    """largest difference between two lists, relative to scale (default: the values themselves)"""
    out = 0.0
    for i, (u, v) in enumerate(zip(a, b)):
        s = scale[i] if scale else max(abs(u), abs(v))
        if s > 0:
            out = max(out, abs(u - v) / s)
    return out


@pytest.mark.parametrize("name", WORKSPACES)
def test_same_results(name):
    command = ["node", str(REPO / "tests/cross/results.mjs"), str(WORKSPACES[name])]
    a = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, cwd=REPO).stdout)
    b = python_results(WORKSPACES[name])
    assert a["parameters"] == b["parameters"]
    assert a["iterations"] == b["iterations"]
    assert a["limit_tests"] == b["limit_tests"]
    assert abs(a["twice_nll"] - b["twice_nll"]) < 1e-9
    unc = [u if u > 0 else 1.0 for u in a["unc"]]
    assert worst(a["pars"], b["pars"], unc) < 1e-9
    assert worst(a["unc"], b["unc"]) < 1e-9
    for key in ["cls", "limit", "z"]:
        assert worst(a[key], b[key]) < 1e-8
    # -2 Δln L of the scans, and the ends of the intervals in units of the uncertainty
    for key in ["scan", "scan2d"]:
        assert len(a[key]) == len(b[key])
        assert worst(a[key], b[key], [1.0] * len(a[key])) < 1e-6
    for u, v in zip(a["intervals"], b["intervals"], strict=True):
        assert u == v if isinstance(u, str) or isinstance(v, str) else abs(u - v) < 1e-5 * a["scan_unc"]


def test_disconnected_intervals():
    """a signal scaled by mu^2 gives a likelihood with two minima: each confidence region is two intervals"""
    path = REPO / "tests/cross/double_minimum.json"
    command = ["node", str(REPO / "tests/cross/scan.mjs"), str(path), "-3", "3", "41"]
    a = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, cwd=REPO).stdout)
    model = histopresto.Model(histopresto.parse_workspace(path.read_text()), 0)
    lik = model.likelihood(model.observed)
    base = infer.fit(model, lik)
    b = [iv.ranges for iv in infer.scan(model, lik, base, model.poi_index, infer.linspace(-3, 3, 41)).intervals]
    for js, py in zip(a, b, strict=True):
        assert len(js) == len(py) == 2
        assert worst([v for r in js for v in r], [v for r in py for v in r]) < 1e-5
        # the two intervals mirror each other
        (lo1, hi1), (lo2, hi2) = py
        assert hi1 < 0 < lo2
        assert abs(lo1 + hi2) < 1e-3 and abs(hi1 + lo2) < 1e-3


def test_infinite_gradient_at_bound():
    """sqrt(theta) has an infinite gradient at theta = 0: a fit started there has to leave the bound"""
    path = REPO / "tests/reference/workspaces/purefunc.json"
    model = histopresto.Model(histopresto.parse_workspace(path.read_text()), 0)
    lik = model.likelihood(model.observed)
    init = list(model.init)
    init[model.names.index("theta")] = 0.0
    command = ["node", str(REPO / "tests/cross/fit.mjs"), str(path), json.dumps(init)]
    a = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, cwd=REPO).stdout)
    b = infer.fit(model, lik, init=init)
    best = infer.fit(model, lik)
    assert a["converged"] and b.status.converged
    assert a["iterations"] == b.status.iterations
    assert abs(a["twice_nll"] - b.twice_nll) < 1e-9
    assert abs(b.twice_nll - best.twice_nll) < 1e-6
    assert worst(a["pars"], b.pars.tolist()) < 1e-6
