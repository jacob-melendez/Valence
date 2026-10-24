// The model files the in-browser engine loads, served under /models/ from the repository's artifacts/ (no copies in
// git): at dev time by a Vite middleware, at build time copied into dist/client/models/ (vite.config.ts).
//
//   /models/core/{m2.txt, m0.txt, model_meta.json, clip_bounds.json}     4-state served models (artifacts/)
//   /models/all/{...same}                                                6-state models (artifacts/all/)
//   /models/emotion/{classifier,valence,arousal}.{txt,json}.gz, meta.json   emotion map (artifacts/emotion/, Cloud B)

import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ARTIFACTS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../artifacts");
const SERVED = ["m2.txt", "m0.txt", "model_meta.json", "clip_bounds.json"];
const EMOTION = [
  "classifier.txt.gz", "valence.txt.gz", "arousal.txt.gz", // LightGBM text models, gzipped (smaller than the dumps)
  "classifier.json.gz", "valence.json.gz", "arousal.json.gz", // booster.dump_model() of the same models
  "meta.json",
  "tour.json.gz", // a scripted 50-minute walk through the 10 states (the camera-free demo, ?demo=tour)
];

/** [url path under /models/, absolute source file] for every model file that exists. */
export function modelFiles() {
  const out = [];
  const add = (dir, sub, names) => {
    for (const n of names) {
      const src = path.join(ARTIFACTS, sub, n);
      if (existsSync(src)) out.push([`${dir}/${n}`, src]);
    }
  };
  add("core", "", SERVED);
  add("all", "all", SERVED);
  add("emotion", "emotion", EMOTION);
  return out;
}

const TYPES = { ".txt": "text/plain; charset=utf-8", ".json": "application/json", ".gz": "application/gzip" };

/** Vite plugin: serve /models/* in dev. */
export function modelFilesPlugin() {
  return {
    name: "valence-model-files",
    configureServer(server) {
      server.middlewares.use("/models/", (req, res, next) => {
        const want = decodeURIComponent((req.url ?? "").split("?")[0].replace(/^\/+/, ""));
        const hit = modelFiles().find(([p]) => p === want);
        if (!hit) return next();
        res.setHeader("Content-Type", TYPES[path.extname(hit[1])] ?? "application/octet-stream");
        res.end(readFileSync(hit[1]));
      });
    },
  };
}

/** Copy the model files into a build output directory (dist/client). */
export function copyModelFiles(outDir = "dist/client") {
  const files = modelFiles();
  for (const [rel, src] of files) {
    const dst = path.join(outDir, "models", rel);
    mkdirSync(path.dirname(dst), { recursive: true });
    copyFileSync(src, dst);
  }
  console.log(`[model-files] ${files.length} files copied to ${path.join(outDir, "models")}`);
}
