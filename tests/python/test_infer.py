"""Edge cases of the inference functions."""

import math
import pathlib

import histopresto
from histopresto import infer

REPO = pathlib.Path(__file__).parents[2]


def test_cls_when_clb_underflows():
    """far above the limit CLs+b and CLb both underflow to zero: CLs is then NaN as in JavaScript, not an error"""
    assert math.isnan(infer._ratio(0.0, 0.0))
    assert infer._ratio(1e-300, 0.0) == math.inf
    assert infer._ratio(1.0, 4.0) == 0.25


def test_yields_without_covariance():
    model = histopresto.Model(histopresto.parse_workspace((REPO / "js/app/examples/search.json").read_text()), 0)
    base = infer.fit(model, model.likelihood(model.observed))
    expected = infer.yields(model, base.pars, base)
    base.cov = None
    out = infer.yields(model, base.pars, base)
    for a, b in zip(out.channels, expected.channels, strict=True):
        assert a.sum == b.sum
        assert math.isnan(a.sum_unc) and all(math.isnan(u) for u in a.total_unc) and all(math.isnan(s.sum_unc) for s in a.samples)
