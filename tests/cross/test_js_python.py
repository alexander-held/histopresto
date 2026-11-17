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
        "intervals": [v if math.isfinite(v) else str(v).replace("inf", "Infinity").replace("nan", "NaN") for iv in scan.intervals for v in (iv.lo, iv.hi)],
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
