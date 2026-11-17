// Scan of the POI of a workspace as JSON on stdout: node scan.mjs <workspace.json> <from> <to> <points>
import fs from "node:fs";
import { Model, infer, num, parseWorkspace } from "@histopresto/core";

const [file, from, to, points] = process.argv.slice(2);
const model = new Model(parseWorkspace(fs.readFileSync(file, "utf8")), 0);
const lik = model.likelihood(model.observed);
const base = await infer.fit(model, lik);
const scan = await infer.scan(model, lik, base, model.poiIndex, num.linspace(Number(from), Number(to), Number(points)));
console.log(JSON.stringify(scan.intervals.map((iv) => iv.ranges)));
