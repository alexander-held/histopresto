"""Combining workspaces: the combination of the two halves of a workspace is the workspace itself."""

import json
import pathlib

import pytest

import histopresto

WORKSPACES = pathlib.Path(__file__).parents[1] / "reference/workspaces"


def read(name):
    return json.loads((WORKSPACES / f"{name}.json").read_text())


def part(ws, keep, name):
    kept = [c["name"] for i, c in enumerate(ws["channels"]) if keep(i)]
    channels = [c for c in ws["channels"] if c["name"] in kept]
    observations = [o for o in ws["observations"] if o["name"] in kept]
    return {"name": name, "workspace": histopresto.parse_workspace({**ws, "channels": channels, "observations": observations})}


@pytest.mark.parametrize("name", ["medium", "purefunc_mixed"])
def test_split_and_combined(name):
    ws = read(name)
    assert len(ws["channels"]) > 1
    whole = histopresto.Model(histopresto.parse_workspace(ws), 0)
    combined = histopresto.combine_workspaces([part(ws, lambda i: i == 0, "a"), part(ws, lambda i: i > 0, "b")])
    assert combined["notes"] == []
    model = histopresto.Model(combined, 0)
    assert list(model.names) == list(whole.names)
    assert list(model.init) == list(whole.init)
    assert model.likelihood(model.observed).value(model.init) == whole.likelihood(whole.observed).value(whole.init)


def test_conflicts():
    ws = read("medium")
    a = part(ws, lambda i: i == 0, "a")
    with pytest.raises(ValueError, match="is in both a and a"):
        histopresto.combine_workspaces([a, a])
    b = part(ws, lambda i: i > 0, "b")
    poi = a["workspace"]["measurements"][0]["config"]["poi"]
    a["workspace"]["measurements"][0]["config"]["parameters"] = [{"name": poi, "inits": [1], "bounds": [[0, 5]]}]
    b["workspace"]["measurements"][0]["config"]["parameters"] = [{"bounds": [[0, 5]], "name": poi, "inits": [1]}, {"name": "other", "fixed": True}]
    assert histopresto.combine_workspaces([a, b])["notes"] == []
    b["workspace"]["measurements"][0]["config"]["parameters"][0]["inits"] = [2]
    combined = histopresto.combine_workspaces([a, b])
    assert len(combined["notes"]) == 1
    assert combined["measurements"][0]["config"]["parameters"] == [{"name": poi, "inits": [1], "bounds": [[0, 5]]}, {"name": "other", "fixed": True}]
