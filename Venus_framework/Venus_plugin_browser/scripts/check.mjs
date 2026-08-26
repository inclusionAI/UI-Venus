import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
if (manifest.manifest_version !== 3) {
  throw new Error("manifest_version must be 3");
}
if (!manifest.permissions.includes("debugger") || !manifest.permissions.includes("sidePanel")) {
  throw new Error("manifest is missing required Chrome permissions");
}

for (const file of walk(root)) {
  if (file.endsWith(".js") || file.endsWith(".mjs")) {
    execFileSync(process.execPath, ["--check", file], { stdio: "inherit" });
  }
}

const prompt = readFileSync(join(root, "prompts", "venus_system.txt"), "utf8");
for (const forbidden of ["box=", "TripleClick", "PressRecent"]) {
  if (prompt.includes(forbidden)) {
    throw new Error(`prompt contains removed schema token: ${forbidden}`);
  }
}

const html = readFileSync(join(root, "sidepanel.html"), "utf8");
if (/(?:src|href)\s*=\s*["']https?:\/\//i.test(html)) {
  throw new Error("sidepanel.html must not load remote assets");
}

console.log("Static extension checks passed.");

function* walk(directory) {
  for (const name of readdirSync(directory)) {
    if (name === "node_modules" || name === ".git") continue;
    const path = join(directory, name);
    if (statSync(path).isDirectory()) yield* walk(path);
    else yield path;
  }
}
