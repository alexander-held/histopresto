// Yields after a fit without covariance matrix: values as usual, uncertainties NaN.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { Model, infer, parseWorkspace } from "@histopresto/core";

test("yields without covariance", async () => {
  const model = new Model(parseWorkspace(fs.readFileSync(new URL("../../js/app/examples/search.json", import.meta.url), "utf8")), 0);
  const base = await infer.fit(model, model.likelihood(model.observed));
  const expected = infer.yields(model, base.pars, base);
  const out = infer.yields(model, base.pars, { ...base, cov: null });
  out.channels.forEach((ch, i) => {
    assert.equal(ch.sum, expected.channels[i].sum);
    assert.ok(Number.isNaN(ch.sumUnc) && ch.totalUnc.every(Number.isNaN) && ch.samples.every((s) => Number.isNaN(s.sumUnc)));
  });
});
