// Fit, CLs, limit, significance and scans of a workspace as JSON on stdout: node results.mjs <workspace.json>
import fs from "node:fs";
import { Model, infer, parseWorkspace } from "@histopresto/core";

const model = new Model(parseWorkspace(fs.readFileSync(process.argv[2], "utf8")), 0);
const lik = model.likelihood(model.observed);
const base = await infer.fit(model, lik);
const test = infer.tester(model, lik, base)(1);
const limit = await infer.limit(model, lik, base);
const significance = await infer.significance(model, lik, base);
// scans of the two parameters with the smallest index that the fit determined
const [p, q] = base.inner;
const range = (i, n) => Array.from({ length: n }, (_, k) => Math.min(Math.max(base.pars[i] + ((k - (n - 1) / 2) * 5 * base.unc[i]) / (n - 1), model.lower[i]), model.upper[i]));
const scan = await infer.scan(model, lik, base, p, range(p, 11));
const scan2d = q === undefined ? null : await infer.scan2d(model, lik, base, [p, q], range(p, 4), range(q, 4));
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
    scan: scan.twiceDeltaNll,
    intervals: scan.intervals.flatMap((iv) => iv.ranges.flat()).map((v) => (Number.isFinite(v) ? v : String(v))),
    scan_unc: base.unc[p],
    scan2d: scan2d ? scan2d.twiceDeltaNll.flat() : [],
  })
);
