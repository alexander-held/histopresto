"""The JavaScript and the Python + C implementation follow the same algorithm step by step: same number of
iterations, results equal up to rounding."""

import json
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
