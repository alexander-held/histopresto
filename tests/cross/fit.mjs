// Fit of a workspace from given starting values as JSON on stdout: node fit.mjs <workspace.json> <init as JSON>
import fs from "node:fs";
import { Model, infer, parseWorkspace } from "@histopresto/core";

const [file, init] = process.argv.slice(2);
const model = new Model(parseWorkspace(fs.readFileSync(file, "utf8")), 0);
const base = await infer.fit(model, model.likelihood(model.observed), { init: JSON.parse(init) });
console.log(JSON.stringify({ twice_nll: base.twiceNll, pars: Array.from(base.pars), iterations: base.status.iterations, converged: base.status.converged }));
