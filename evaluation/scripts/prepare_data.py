#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# prepare_data.py — convert official datasets into the evaluation data layout.
#
# Subcommands:
# osworld: convert the official benchmark/ directory:
#   os-world-g-images/ copies the official images.
#   sspro_annotations/osworld_g*.json converts OSWorld-G_refined.json.
#   os-world-g-refine/annotation.json extracts metadata from the same source.
# sspro: copy official annotations/images and generate fixed debug subsets.
#             test_10.json / test_50.json
# hrbench: official hr_bench_{4k,8k}.tsv -> manifest and content-addressed images.
#   Uses the same conversion as hrbench/eval_hrbench_agentic.py.
#
# Conversion rules are checked against the release data reference:
# OSWorld bbox: floating-point [x,y,w,h] -> integer [x1,y1,x2,y2], truncating with int.
# OSWorld polygon: enclosing rectangle [int(min_x),int(min_y),int(max_x),int(max_y)].
#   OSWorld-G refusal: bbox=[-1,-1,-1,-1], data_type='refusal'
#
# See scripts/fetch_data.sh or each subcommand --help for usage.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# OS-World-G conversion.
# ---------------------------------------------------------------------------
# Fixed workflow subsets are available for additional experiments, not the standard run.
WF_IDS = [
    "osworld_g_0001", "osworld_g_0009", "osworld_g_0013", "osworld_g_0018",
    "osworld_g_0061", "osworld_g_0077", "osworld_g_0078", "osworld_g_0092",
    "osworld_g_0095", "osworld_g_0098", "osworld_g_0108", "osworld_g_0109",
    "osworld_g_0111", "osworld_g_0128", "osworld_g_0132", "osworld_g_0134",
    "osworld_g_0163", "osworld_g_0203", "osworld_g_0215", "osworld_g_0241",
    "osworld_g_0249", "osworld_g_0289", "osworld_g_0292", "osworld_g_0299",
    "osworld_g_0318", "osworld_g_0322", "osworld_g_0326", "osworld_g_0337",
    "osworld_g_0340", "osworld_g_0346", "osworld_g_0367", "osworld_g_0373",
    "osworld_g_0383", "osworld_g_0392", "osworld_g_0405", "osworld_g_0407",
    "osworld_g_0428", "osworld_g_0436", "osworld_g_0444", "osworld_g_0445",
    "osworld_g_0448", "osworld_g_0458", "osworld_g_0460", "osworld_g_0474",
    "osworld_g_0484", "osworld_g_0487", "osworld_g_0491", "osworld_g_0501",
    "osworld_g_0507",
]
WF_REFUSAL_IDS = [
    "osworld_g_0517", "osworld_g_0526", "osworld_g_0537", "osworld_g_0552",
    "osworld_g_0555", "osworld_g_0561", "osworld_g_0563",
]


def osworld_bbox_convert(box_type: str, coordinates: list) -> list[int]:
    """Convert official bbox [x,y,w,h] or polygon points to an integer enclosing bbox."""
    if box_type == "bbox":
        x, y, w, h = coordinates
        return [int(x), int(y), int(x + w), int(y + h)]
    if box_type == "polygon":
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    if box_type == "refusal":
        return [-1, -1, -1, -1]
    raise ValueError(f"unexpected box_type: {box_type}")


def prepare_osworld(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.official_benchmark).resolve()
    refined_path = benchmark_dir / "OSWorld-G_refined.json"
    with refined_path.open(encoding="utf-8") as handle:
        refined = json.load(handle)

    out_root = Path(args.data_root)
    ann_dir = out_root / "osworld_g" / "sspro_annotations"
    refine_dir = out_root / "osworld_g" / "os-world-g-refine"
    images_dir = out_root / "osworld_g" / "os-world-g-images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    refine_dir.mkdir(parents=True, exist_ok=True)

    # 1) Annotation IDs follow the official refined ordering.
    pos_rows: list[dict] = []
    refusal_rows: list[dict] = []
    refine_rows: list[dict] = []
    for idx, item in enumerate(refined):
        kit_bbox = osworld_bbox_convert(item["box_type"], item["box_coordinates"])
        prefix = "refusal" if item["box_type"] == "refusal" else "annotation"
        # Readable prefixes do not alter the global osworld_g_{i:04d} ID ordering.
        entry = {
            "img_filename": item["image_path"],
            "bbox": kit_bbox,
            "instruction": item["instruction"],
            "instruction_cn": None,
            "id": f"osworld_g_{idx:04d}",
            "application": "mixed",
            "platform": "desktop",
            "img_size": list(item["image_size"]),
            "ui_type": "unknown",
            "group": "osworld_g",
            "data_type": item["box_type"],
        }
        refine_rows.append(
            {
                "img_filename": item["image_path"],
                "bbox": kit_bbox,
                "instruction": item["instruction"],
                "data_type": item["box_type"],
            }
        )
        if item["box_type"] == "refusal":
            refusal_rows.append(entry)
        else:
            pos_rows.append(entry)

    def dump_json(path: Path, value, *, indent: int, ensure_ascii: bool) -> None:
        path.write_text(
            json.dumps(value, indent=indent, ensure_ascii=ensure_ascii),
            encoding="utf-8",
        )

    dump_json(ann_dir / "osworld_g.json", pos_rows, indent=1, ensure_ascii=False)
    dump_json(ann_dir / "osworld_g_refusal.json", refusal_rows, indent=1, ensure_ascii=True)
    wf = [row for row in pos_rows if row["id"] in set(WF_IDS)]
    wf_refusal = [row for row in refusal_rows if row["id"] in set(WF_REFUSAL_IDS)]
    if len(wf) != len(WF_IDS) or len(wf_refusal) != len(WF_REFUSAL_IDS):
        raise SystemExit(
            f"wf subset mismatch: got {len(wf)}/{len(WF_IDS)} pos, "
            f"{len(wf_refusal)}/{len(WF_REFUSAL_IDS)} refusal (official dataset version changed?)"
        )
    dump_json(ann_dir / "osworld_g_wf.json", wf, indent=1, ensure_ascii=True)
    dump_json(ann_dir / "osworld_g_wf_refusal.json", wf_refusal, indent=1, ensure_ascii=True)

    # ---- 2) os-world-g-refine/annotation.json ----
    dump_json(refine_dir / "annotation.json", refine_rows, indent=4, ensure_ascii=False)

    # 3) Copy all official benchmark images idempotently, including script.py.
    src_images = benchmark_dir / "images"
    if not src_images.is_dir():
        raise SystemExit(f"missing official images dir: {src_images}")
    copied = 0
    for src_file in sorted(src_images.rglob("*")):
        if not src_file.is_file():
            continue
        dst_file = images_dir / src_file.relative_to(src_images)
        if dst_file.is_file():
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_file, dst_file)
        copied += 1
    print(f"osworld: copied {copied} files -> {images_dir}")

    print(
        f"osworld: pos={len(pos_rows)} refusal={len(refusal_rows)} refine={len(refine_rows)}"
    )


# ---------------------------------------------------------------------------
# ScreenSpot-Pro layout and debug subsets.
# ---------------------------------------------------------------------------
# Fixed debug IDs: test_10 and test_50 are disjoint subsets of official annotations.
SSPRO_TEST_10_IDS = [
    "illustrator_windows_0", "illustrator_windows_3", "illustrator_windows_4",
    "illustrator_windows_7", "illustrator_windows_8", "illustrator_windows_20",
    "illustrator_windows_21", "illustrator_windows_23", "illustrator_windows_25",
    "illustrator_windows_29",
]
SSPRO_TEST_50_IDS = [
    "vivado_windows_69", "davinci_macos_43", "android_studio_macos_51",
    "word_macos_21", "macos_common_macos_12", "linux_common_linux_0",
    "inventor_windows_26", "excel_macos_6", "word_macos_11", "davinci_macos_24",
    "vscode_macos_24", "word_macos_19", "solidworks_windows_37", "blender_windows_64",
    "unreal_engine_windows_4", "powerpoint_windows_42", "android_studio_macos_65",
    "android_studio_macos_61", "davinci_macos_6", "inventor_windows_16",
    "inventor_windows_45", "quartus_windows_0", "unreal_engine_windows_27",
    "android_studio_macos_54", "solidworks_windows_70", "illustrator_windows_7",
    "windows_common_windows_50", "vmware_macos_10", "windows_common_windows_20",
    "powerpoint_windows_37", "inventor_windows_20", "premiere_windows_15",
    "unreal_engine_windows_1", "macos_common_macos_18", "android_studio_macos_13",
    "word_macos_57", "excel_macos_47", "windows_common_windows_13",
    "powerpoint_windows_43", "matlab_macos_80", "excel_macos_39", "inventor_windows_9",
    "word_macos_66", "matlab_macos_73", "davinci_macos_4", "photoshop_windows_7",
    "davinci_macos_13", "origin_windows_26", "matlab_macos_88", "unreal_engine_windows_31",
]


def prepare_sspro(args: argparse.Namespace) -> None:
    official_root = Path(args.official_sspro).resolve()
    official_ann = official_root / "annotations"
    official_images = official_root / "images"
    out_root = Path(args.data_root)
    target = out_root / "sspro" / "Screenspot-pro"
    ann_dir = target / "annotations"
    image_dir = target / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy official annotations, excluding non-official debug files.
    copied = 0
    for src in sorted(official_ann.glob("*.json")):
        if src.name in ("test_10.json", "test_50.json"):
            continue
        shutil.copyfile(src, ann_dir / src.name)
        copied += 1
    print(f"sspro: copied {copied} official annotation files")

    # 2) Build the debug subsets using the embedded IDs.
    pool: dict[str, dict] = {}
    for src in sorted(official_ann.glob("*.json")):
        if src.name in ("test_10.json", "test_50.json"):
            continue
        for row in json.loads(src.read_text(encoding="utf-8")):
            pool[row["id"]] = row

    def subset(path: Path, ids: list[str], *, ensure_ascii: bool) -> None:
        missing = [i for i in ids if i not in pool]
        if missing:
            raise SystemExit(f"sspro: subset ids missing in official data: {missing[:5]}")
        value = [pool[i] for i in ids]
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=ensure_ascii), encoding="utf-8"
        )

    subset(ann_dir / "test_10.json", SSPRO_TEST_10_IDS, ensure_ascii=True)
    subset(ann_dir / "test_50.json", SSPRO_TEST_50_IDS, ensure_ascii=False)
    print(f"sspro: wrote test_10.json ({len(SSPRO_TEST_10_IDS)}) test_50.json ({len(SSPRO_TEST_50_IDS)})")

    # 3) Copy official images, excluding .DS_Store.
    if not official_images.is_dir():
        raise SystemExit(f"sspro: missing official images dir: {official_images}")
    n = 0
    for src in official_images.rglob("*.png"):
        rel = src.relative_to(official_images)
        dst = image_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
            n += 1
    print(f"sspro: copied {n} images -> {image_dir}")


# ---------------------------------------------------------------------------
# HR-Bench: TSV -> manifest + content-addressed images
# ---------------------------------------------------------------------------
CHOICES = ("A", "B", "C", "D")


def image_extension_and_size(raw: bytes) -> tuple[str, list[int]]:
    with Image.open(BytesIO(raw)) as image:
        image.load()
        image_format = (image.format or "png").lower()
        extension = {"jpeg": "jpg", "tiff": "tif"}.get(image_format, image_format)
        return extension, [int(image.width), int(image.height)]


def prepare_hrbench(args: argparse.Namespace) -> None:
    tsv_dir = Path(args.official_tsv_dir).resolve()
    out_cache = Path(args.data_root) / "hrbench"
    images_dir = out_cache / "images"
    out_cache.mkdir(parents=True, exist_ok=True)

    csv.field_size_limit(sys.maxsize)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for split in splits:
        tsv_path = tsv_dir / f"hr_bench_{split}.tsv"
        if not tsv_path.is_file():
            raise SystemExit(f"hrbench: missing TSV {tsv_path}")
        tasks: list[dict] = []
        active_group: int | None = None
        active_encoded: str | None = None
        active_name = ""
        active_size: list[int] = [0, 0]
        seen_groups = 0
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"index", "answer", "question", "category", "cycle_category", "image", *CHOICES}
            missing_cols = required - set(reader.fieldnames or [])
            if missing_cols:
                raise SystemExit(f"hrbench: missing columns {sorted(missing_cols)}")
            for row in reader:
                index = int(row["index"])
                group_index = index // 4
                if group_index != active_group:
                    active_group = group_index
                    active_encoded = row["image"]
                    seen_groups += 1
                    raw = base64.b64decode(active_encoded)
                    digest = hashlib.sha256(raw).hexdigest()
                    extension, size = image_extension_and_size(raw)
                    rel_name = f"data/hrbench/images/{digest}.{extension}"
                    dst = out_cache / "images" / f"{digest}.{extension}"
                    if not dst.is_file():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(raw)
                    active_name = rel_name
                    active_size = size
                elif row["image"] != active_encoded:
                    raise SystemExit(f"hrbench: group {group_index} changes image")
                tasks.append(
                    {
                        "eval_key": f"hrbench{split}:{index}",
                        "order": index,
                        "index": index,
                        "group_index": group_index,
                        "split": split,
                        "answer": row["answer"].strip().upper(),
                        "question": row["question"].strip(),
                        "options": {choice: row[choice] for choice in CHOICES},
                        "category": row["category"].strip(),
                        "cycle_category": int(row["cycle_category"]),
                        "image_path": active_name,
                        "img_size": list(active_size),
                    }
                )
        manifest_path = out_cache / f"hrbench_{split}_manifest.json"
        stat = tsv_path.stat()
        manifest = {
            "version": 1,
            "source": {
                "path": str(tsv_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "split": split,
            "complete": True,
            "tasks": tasks,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"hrbench {split}: tasks={len(tasks)} groups={seen_groups} -> {manifest_path}"
        )


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="benchmark", required=True)

    p_osw = sub.add_parser("osworld", help="Official OSWorld-G benchmark/ -> evaluation data/")
    p_osw.add_argument("--official_benchmark", required=True,
                       help="Official benchmark/ directory containing OSWorld-G_refined.json and images/")
    p_osw.add_argument("--data_root", default="data")

    p_ss = sub.add_parser("sspro", help="Official ScreenSpot-Pro -> evaluation data/ with debug subsets")
    p_ss.add_argument("--official_sspro", required=True,
                      help="Official ScreenSpot-Pro root containing annotations/ and images/")
    p_ss.add_argument("--data_root", default="data")

    p_hb = sub.add_parser("hrbench", help="Official HR-Bench TSV -> manifest and images")
    p_hb.add_argument("--official_tsv_dir", required=True,
                      help="Directory containing hr_bench_4k.tsv and hr_bench_8k.tsv")
    p_hb.add_argument("--data_root", default="data")
    p_hb.add_argument("--splits", default="4k,8k",
                      help="Comma-separated splits to prepare (default: 4k,8k)")

    args = parser.parse_args()
    if args.benchmark == "osworld":
        prepare_osworld(args)
    elif args.benchmark == "sspro":
        prepare_sspro(args)
    elif args.benchmark == "hrbench":
        prepare_hrbench(args)


if __name__ == "__main__":
    main()
