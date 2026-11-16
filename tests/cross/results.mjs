// Fit, CLs, limit and significance of a workspace as JSON on stdout: node results.mjs <workspace.json>
import fs from "node:fs";
import { Model, infer, parseWorkspace } from "@histopresto/core";

const model = new Model(parseWorkspace(fs.readFileSync(process.argv[2], "utf8")), 0);
const lik = model.likelihood(model.observed);
const base = await infer.fit(model, lik);
const test = infer.tester(model, lik, base)(1);
const limit = await infer.limit(model, lik, base);
const significance = await infer.significance(model, lik, base);
console.log(
  JSON.stringify({
    parameters: model.names,
    iterations: base.status.iterations,
    twice_nll: base.twiceNll,
    pars: Array.from(base.pars),
    unc: Array.from(base.unc),
    cls: [test.cls, ...test.expected],
    limit: [limit.observed, ...limit.expected],
    limit_tests: limit.results.length,
    z: [significance.z, significance.zExpected],
  })
);
