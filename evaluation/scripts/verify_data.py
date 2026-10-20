#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# verify_data.py — verify benchmark data against the release reference.
#
# Byte-level reference: scripts/data_manifest.sha256 (2213 entries).
# Includes SSPro, OS-World-G, and PerceptionBench images/annotations,
# plus HR-Bench images and two manifests.
# HR-Bench source paths and timestamps depend on the build environment.
# Compare their tasks to the reference bundles instead of file bytes:
# scripts/data_manifest_hrbench_tasks_{4k,8k}.json, including source.size.
#
# Usage: python3 scripts/verify_data.py [--data-root data] [--quiet]
# Exit status: 0 = all checks passed; 1 = differences found.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data_manifest.sha256"
HRBENCH_REFERENCE = {
    "4k": Path(__file__).resolve().parent / "data_manifest_hrbench_tasks_4k.json",
    "8k": Path(__file__).resolve().parent / "data_manifest_hrbench_tasks_8k.json",
}
# Manifests checked by content; paths are relative to data-root.
CONTENT_LEVEL_FILES = {
    "./hrbench/hrbench_4k_manifest.json": "4k",
    "./hrbench/hrbench_8k_manifest.json": "8k",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        entries[rel.strip().removeprefix("./")] = digest
    return entries


def verify_content_level(data_root: Path, quiet: bool) -> list[str]:
    problems: list[str] = []
    for rel, split in CONTENT_LEVEL_FILES.items():
        path = data_root / rel.removeprefix("./")
        reference_path = HRBENCH_REFERENCE[split]
        if not path.is_file():
            problems.append(f"MISSING(content): {rel}")
            continue
        if not reference_path.is_file():
            problems.append(f"missing reference bundle: {reference_path}")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if manifest.get("tasks") != reference.get("tasks"):
            for i, (a, b) in enumerate(
                zip(manifest.get("tasks") or [], reference.get("tasks") or [])
            ):
                if a != b:
                    problems.append(
                        f"DIFF(content): {rel} tasks[{i}] mismatch "
                        f"(first field: { {k: a.get(k) for k in sorted(set(a) ^ set(b)) if k in a} if a else {} })"
                    )
                    break
            else:
                problems.append(
                    f"DIFF(content): {rel} task count "
                    f"{len(manifest.get('tasks') or [])} != {len(reference.get('tasks') or [])}"
                )
        elif manifest.get("source", {}).get("size") != reference.get("source_size"):
            problems.append(
                f"DIFF(content): {rel} source.size "
                f"{manifest.get('source', {}).get('size')} != {reference.get('source_size')}"
            )
        elif manifest.get("split") != split or manifest.get("complete") is not True:
            problems.append(f"DIFF(content): {rel} split/complete mismatch")
        else:
            if not quiet:
                print(f"  OK(content)  {rel}  tasks={len(manifest['tasks'])}")
    return problems


def verify_byte_level(
    data_root: Path, entries: dict[str, str], quiet: bool
) -> tuple[list[str], dict[str, int]]:
    problems: list[str] = []
    stats: dict[str, int] = {"checked": 0, "ok": 0, "missing": 0, "changed": 0}
    for rel, digest in sorted(entries.items()):
        if f"./{rel}" in CONTENT_LEVEL_FILES:
            continue
        path = data_root / rel
        stats["checked"] += 1
        if not path.is_file():
            stats["missing"] += 1
            problems.append(f"MISSING: {rel}")
            continue
        if sha256_of(path) == digest:
            stats["ok"] += 1
        else:
            stats["changed"] += 1
            problems.append(f"DIFF: {rel}")
        if not quiet and stats["checked"] % 400 == 0:
            print(f"  ... {stats['checked']}/{len(entries)}", flush=True)
    return problems, stats


def collect_existing(data_root: Path) -> set[str]:
    return {
        str(path.relative_to(data_root))
        for path in data_root.rglob("*")
        if path.is_file()
        if path.name != ".DS_Store"
        if not path.name.startswith(".hrbench_")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    entries = load_manifest(manifest_path)
    print(f"Reference: {manifest_path} ({len(entries)} files)")

    problems, stats = verify_byte_level(data_root, entries, args.quiet)
    problems += verify_content_level(data_root, args.quiet)

    existing = collect_existing(data_root)
    known = set(entries)
    extra = sorted(existing - known)
    for rel in extra:
        show = f"EXTRA: {rel}"
        problems.append(show)

    print(
        f"\nbyte-level: checked={stats['checked']} ok={stats['ok']} "
        f"missing={stats['missing']} changed={stats['changed']}"
    )
    print(f"content-level: 2 manifests vs reference bundles")
    print(f"extra files in data/: {len(extra)}")

    per_benchmark: dict[str, int] = {}
    for rel in sorted(existing):
        top = rel.split("/", 1)[0]
        per_benchmark[top] = per_benchmark.get(top, 0) + 1
    print("Existing files by dataset:", json.dumps(per_benchmark, ensure_ascii=False))

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for line in problems[:40]:
            print(" ", line)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        sys.exit(1)
    print("\nALL OK — data matches the release reference")


if __name__ == "__main__":
    main()
