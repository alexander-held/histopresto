"""Edge cases of the inference functions."""

import math
import pathlib

import histopresto
from histopresto import infer

REPO = pathlib.Path(__file__).parents[2]


def test_yields_without_covariance():
    model = histopresto.Model(histopresto.parse_workspace((REPO / "js/app/examples/search.json").read_text()), 0)
    base = infer.fit(model, model.likelihood(model.observed))
    expected = infer.yields(model, base.pars, base)
    base.cov = None
    out = infer.yields(model, base.pars, base)
    for a, b in zip(out.channels, expected.channels, strict=True):
        assert a.sum == b.sum
        assert math.isnan(a.sum_unc) and all(math.isnan(u) for u in a.total_unc) and all(math.isnan(s.sum_unc) for s in a.samples)
