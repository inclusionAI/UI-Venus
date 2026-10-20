"""Shared prediction filenames; keep existing names for checkpoint compatibility."""
from pathlib import Path


def shard_tag(num_shards, shard_index):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("NUM_SHARDS must be positive and SHARD_INDEX in [0, NUM_SHARDS)")
    return f"_shard{shard_index}of{num_shards}" if num_shards > 1 else ""


def prediction_prefix(model, output_suffix=""):
    if "/" in output_suffix or "\\" in output_suffix:
        raise ValueError("PB_OUTPUT_SUFFIX must be a filename suffix, without path separators")
    return f"PerceptionBench_agentic_{model.replace('/', '_')}{output_suffix}"


def prediction_basename(model, num_shards, shard_index, output_suffix=""):
    return prediction_prefix(model, output_suffix) + shard_tag(num_shards, shard_index)


def discover_predictions(directories, prefix, num_shards):
    """Resolve one whole set; never fall back past a partial or ambiguous run."""
    shard_tag(num_shards, 0)
    for directory in dict.fromkeys(Path(p) for p in directories):
        paths = [directory / f"{prefix}{shard_tag(num_shards, i)}.jsonl"
                 for i in range(num_shards)]
        if num_shards == 1:
            # Accept the old finalizer's spelling as a read-only compatibility alias.
            candidates = [paths[0], directory / f"{prefix}_shard0of1.jsonl"]
            found = [p for p in candidates if p.is_file()]
            if len(found) > 1:
                raise ValueError(f"Ambiguous single-shard inputs in {directory}: {found}")
            if found:
                return found
        else:
            found = [p for p in paths if p.is_file()]
            if found and len(found) != num_shards:
                raise ValueError(f"Incomplete shard set in {directory}: found {len(found)}/{num_shards}")
            if found:
                return paths
    raise FileNotFoundError(f"No complete prediction set for {prefix}, num_shards={num_shards}")
