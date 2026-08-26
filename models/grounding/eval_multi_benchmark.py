"""
eval_multi_benchmark.py
=======================

Multi-benchmark grounding evaluation driver.

For each benchmark, it spawns an eval_single_benchmark.py subprocess,
reusing its direct inference, concurrency, and checkpoint/resume logic.

Supported benchmarks:
  - screenspot_pro:   ScreenSpot-Pro (default)
  - venusbench_gd:    VenusBench-GD
  - osworld_g_refine: OSWorld-G-Refine
  - ss_v2_resize:     SS_v2_resize
  - mmbench_gui:      MMBench-GUI-OfflineImages
  - data_uivision:    UI-Vision element grounding
  - ui_vision_full:   UI-Vision Full (Element + Layout)

Usage:

  1. Evaluate all benchmarks:
     python3 models/grounding/eval_multi_benchmark.py \
         --base_url http://127.0.0.1:8011/v1 \
         --api_key empty \
         --model_name qwen3.6-model \
         --benchmarks all \
         --num_workers 4 \
         --log_dir results/exp01 \
         --checkpoint_dir results_mid/exp01

  2. Evaluate specific benchmarks only:
     python3 models/grounding/eval_multi_benchmark.py \
         --base_url http://127.0.0.1:8011/v1 \
         --api_key empty \
         --model_name qwen3.6-model \
         --benchmarks screenspot_pro,venusbench_gd,osworld_g_refine,ss_v2_resize \
         --num_workers 8 \
         --log_dir results/ui-venus2_9b_50steps_from1k \
         --checkpoint_dir results_mid/ui-venus2_9b_50steps_from1k

  3. Run multi-process evaluation with num_workers threads per process:
     python3 models/grounding/eval_multi_benchmark.py \
         --base_url http://127.0.0.1:8011/v1 \
         --api_key empty \
         --model_name qwen3.6-model \
         --benchmarks all \
         --num_processes 4 \
         --num_workers 4 \
         --log_dir results/exp01 \
         --checkpoint_dir results_mid/exp01

Output:
  Generated under --log_dir:
    - <benchmark>.json          Full per-benchmark results
    - summary.json              Accuracy summary for all benchmarks

  Generated under --checkpoint_dir:
    - <benchmark>_ckpt.json     Resumable checkpoint files
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Benchmark data paths
#   Key: benchmark identifier
#   Value: {"imgs": image directory, "annotations": annotation JSON directory, "description": description}
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = {
    "screenspot_pro": {
        "imgs": "/path/to/Screenspot-pro/images",
        "annotations": "/path/to/Screenspot-pro/annotations",
        "description": "ScreenSpot-Pro",
    },
    "venusbench_gd": {
        "imgs": "/path/to/VenusBench-GD/images/",
        "annotations": "/path/to/VenusBench-GD/instruction/",
        "description": "VenusBench-GD",
    },
    "osworld_g_refine": {
        "imgs": "/path/to/os-world-g-images/",
        "annotations": "/path/to/osworld-g-refine/",
        "description": "OSWorld-G-Refine",
    },
    "ss_v2_resize": {
        "imgs": "/path/to/SS_v2_resize/screenspotv2_image/",
        "annotations": "/path/to/SS_v2_resize/json_file/",
        "description": "SS_v2_resize",
    },
    "mmbench_gui": {
        "imgs": "/path/to/MMBench-GUI-OfflineImages/",
        "annotations": "/path/to/MMBench-GUI-OfflineImages/json/",
        "description": "MMBench-GUI-OfflineImages",
    },
    "data_uivision": {
        "imgs": "/path/to/data_uivision/images/",
        "annotations": "/path/to/data_uivision/annotations/element_grounding/",
        "description": "UI-Vision Element Grounding",
    },
    "ui_vision_full": {
        "imgs": "/path/to/data_uivision/images/",
        "annotations": "/path/to/data_uivision/annotations/",
        "description": "UI-Vision Full (Element + Layout)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-benchmark grounding evaluation (drives eval_single_benchmark.py)."
    )

    # --- Benchmark selection ---
    parser.add_argument('--benchmarks', type=str, default='all',
                        help="逗号分隔的 benchmark 列表, 或 'all' 表示全部。"
                             f"可选: {', '.join(BENCHMARK_CONFIGS.keys())}")

    # --- Output paths ---
    parser.add_argument('--log_dir', type=str, required=True,
                        help="日志输出目录 (每个 benchmark 一个子文件)")
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help="Checkpoint 保存目录 (每个 benchmark 一个子文件)")

    # --- API endpoint configuration ---
    parser.add_argument('--base_url', type=str, default='http://127.0.0.1/v1')
    parser.add_argument('--api_key', type=str, default='empty')
    parser.add_argument('--model_name', type=str, default='qwen3.6-model')

    # --- Inference configuration ---
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=20)
    parser.add_argument('--norm_type', type=str, default='0-1000',
                        choices=['pixel', '0-1000'])

    # --- Custom prompts ---
    parser.add_argument('--system_prompt', type=str, default=None)
    parser.add_argument('--user_prompt', type=str, default=None)

    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def _build_subprocess_command(args, benchmark_name, config):
    """Build the argument list for an eval_single_benchmark.py subprocess."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "eval_single_benchmark.py"),
        "--base_url", args.base_url,
        "--api_key", args.api_key,
        "--model_name", args.model_name,
        "--task", "all",
        "--language", "en",
        "--gt_type", "positive",
        "--inst_style", "instruction",
        "--num_workers", str(args.num_workers),
        "--num_processes", str(args.num_processes),
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--norm_type", args.norm_type,
        "--screenspot_imgs", config["imgs"],
        "--screenspot_test", config["annotations"],
    ]

    # ui_vision_full scans both element_grounding and layout_grounding subdirectories.
    if benchmark_name == "ui_vision_full":
        cmd += ["--task", "element_grounding,layout_grounding"]

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{benchmark_name}.json")
    cmd += ["--log_path", log_path]

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, f"{benchmark_name}_ckpt.json")
        cmd += ["--checkpoint_path", ckpt_path]

    if args.system_prompt is not None:
        cmd += ["--system_prompt", args.system_prompt]
    if args.user_prompt is not None:
        cmd += ["--user_prompt", args.user_prompt]

    if args.debug:
        cmd += ["--debug"]

    return cmd


def run_benchmark(args, benchmark_name, config):
    import subprocess

    print(f"\n{'='*70}")
    print(f"  Benchmark: {config['description']} ({benchmark_name})")
    print(f"  Images:    {config['imgs']}")
    print(f"  Annotations: {config['annotations']}")
    print(f"{'='*70}\n")

    cmd = _build_subprocess_command(args, benchmark_name, config)

    if not os.path.isdir(config["imgs"]):
        print(f"[WARN] Image directory does not exist, skipping: {config['imgs']}")
        return None
    if not os.path.isdir(config["annotations"]):
        print(f"[WARN] Annotation directory does not exist, skipping: {config['annotations']}")
        return None

    display_cmd = cmd.copy()
    display_cmd[display_cmd.index("--api_key") + 1] = "***"
    print(f"[CMD] {' '.join(display_cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Benchmark {benchmark_name} failed with return code {result.returncode}")
        return None

    log_path = os.path.join(args.log_dir, f"{benchmark_name}.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            report = json.load(f)
        overall = report.get("metrics", {}).get("overall", {})
        print(f"\n[RESULT] {benchmark_name}: "
              f"action_acc={overall.get('action_acc', 'N/A'):.4f}, "
              f"text_acc={overall.get('text_acc', 'N/A'):.4f}, "
              f"icon_acc={overall.get('icon_acc', 'N/A'):.4f}, "
              f"total={overall.get('num_total', 'N/A')}")
        return overall
    return None


def print_summary(results_summary, args):
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: {args.model_name}")
    print(f"{'='*80}")
    print(f"{'Benchmark':<25} {'ActionAcc':>10} {'TextAcc':>10} {'IconAcc':>10} {'Total':>8}")
    print(f"{'-'*80}")
    for name, metrics in results_summary.items():
        desc = BENCHMARK_CONFIGS[name]['description']
        print(f"{desc:<25} {metrics.get('action_acc', 0):>10.4f} "
              f"{metrics.get('text_acc', 0):>10.4f} "
              f"{metrics.get('icon_acc', 0):>10.4f} "
              f"{metrics.get('num_total', 0):>8}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()

    if args.benchmarks == "all":
        benchmark_names = list(BENCHMARK_CONFIGS.keys())
    else:
        benchmark_names = [b.strip() for b in args.benchmarks.split(",")]
        invalid = set(benchmark_names) - set(BENCHMARK_CONFIGS.keys())
        if invalid:
            print(f"[ERROR] Unknown benchmarks: {invalid}")
            print(f"Available: {list(BENCHMARK_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Will evaluate {len(benchmark_names)} benchmark(s): {benchmark_names}")

    results_summary = {}
    failed_benchmarks = []
    for name in benchmark_names:
        result = run_benchmark(args, name, BENCHMARK_CONFIGS[name])
        if result is not None:
            results_summary[name] = result
        else:
            failed_benchmarks.append(name)

    if len(results_summary) > 1:
        print_summary(results_summary, args)

        summary_path = os.path.join(args.log_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "model_name": args.model_name,
                "benchmarks": results_summary,
            }, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_path}")

    if failed_benchmarks:
        print(f"[ERROR] Benchmarks not completed: {failed_benchmarks}")
        sys.exit(1)


if __name__ == "__main__":
    main()
