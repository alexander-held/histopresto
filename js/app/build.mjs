// Builds the site into dist/: two classic-script bundles (page and worker), the static files and the examples.
// Classic scripts rather than modules so that dist/index.html also works when opened from file://.
// Usage: node build.mjs [--serve]
import fs from "node:fs";
import path from "node:path";
import * as esbuild from "esbuild";

const here = import.meta.dirname;
const dist = path.join(here, "dist");
fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(path.join(dist, "examples"), { recursive: true });

for (const file of ["index.html", "style.css"]) fs.copyFileSync(path.join(here, file), path.join(dist, file));
for (const file of ["icon.svg", "icon-dark.svg"]) fs.copyFileSync(path.join(here, "../../assets", file), path.join(dist, file));

// examples are scripts loaded on demand (fetch does not work from file://)
const examples = JSON.parse(fs.readFileSync(path.join(here, "examples/index.json"), "utf8"));
for (const { id, file } of examples) {
  const workspace = JSON.parse(fs.readFileSync(path.join(here, "examples", file), "utf8"));
  fs.writeFileSync(path.join(dist, "examples", `${id}.js`), `HF_EXAMPLE(${JSON.stringify(id)}, ${JSON.stringify(workspace)});\n`);
}
const listing = examples.map(({ id, title, note }) => ({ id, title, note }));
fs.writeFileSync(path.join(dist, "examples/index.js"), `window.HF_EXAMPLES = ${JSON.stringify(listing, null, 1)};\n`);

const options = {
  entryPoints: { app: path.join(here, "src/main.js"), worker: path.join(here, "src/worker.js") },
  bundle: true,
  format: "iife",
  target: "es2020",
  outdir: dist,
  logLevel: "info",
};
if (process.argv.includes("--serve")) {
  const context = await esbuild.context(options);
  await context.watch();
  const { port } = await context.serve({ servedir: dist, port: 8000 });
  console.log(`http://localhost:${port}/`);
} else {
  await esbuild.build(options);
}
