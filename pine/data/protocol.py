"""Dependency-free split isolation and query-time support selection."""
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "msg", "valid", "test")
PROTOCOL = "pine_count_preserved_v2"


def read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fin:
        for number, line in enumerate(fin, 1):
            parts = line.rstrip().split("\t")
            if len(parts) not in (4, 5):
                raise ValueError(f"{path}:{number}: expected 4 or 5 columns")
            row = tuple(map(int, parts[:4]))
            if min(row) < 0:
                raise ValueError(f"{path}:{number}: negative ID")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def read_splits(data_dir):
    return {split: read_rows(Path(data_dir) / f"{split}.txt") for split in SPLITS}


def validate_splits(splits):
    """Train/support may overlap; evaluation targets must be held out."""
    for split in SPLITS:
        if not splits[split]:
            raise ValueError(f"Empty {split} split")
        if len(splits[split]) != len(set(splits[split])):
            raise ValueError(f"Duplicate quadruples within {split}")
    for a, b in (("train", "valid"), ("train", "test"), ("msg", "valid"),
                 ("msg", "test"), ("valid", "test")):
        overlap = set(splits[a]) & set(splits[b])
        if overlap:
            raise ValueError(
                f"{a}/{b} overlap: {len(overlap)} quadruples, example={min(overlap)}. "
                "Remove held-out targets from training and support before preprocessing."
            )


def source_fingerprints(data_dir):
    data_dir = Path(data_dir)
    names = [f"{split}.txt" for split in SPLITS]
    names += [name for name in ("entity2id.txt", "rel2id.txt", "relation2id.txt", "time2id.txt")
              if (data_dir / name).is_file()]
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in names}


def validate_processed(data_dir, processed_dir, setting):
    """Reject artifacts built from old splits or old support-selection rules."""
    marker_path = Path(processed_dir) / "storage_format.json"
    if not marker_path.is_file():
        raise ValueError("Missing verified storage_format.json; rerun preprocessing.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("protocol") != PROTOCOL or marker.get("setting") != setting:
        raise ValueError(
            f"Old or incompatible preprocessing protocol in {marker_path}: "
            f"found protocol={marker.get('protocol')!r}, setting={marker.get('setting')!r}; "
            f"expected protocol={PROTOCOL!r}, setting={setting!r}. "
            "Rebuild with python -m pine.preprocess in a new processed directory."
        )
    if marker.get("source_sha256") != source_fingerprints(data_dir):
        raise ValueError("Processed graphs do not match the current split/mapping files; rerun preprocessing.")
    if setting == "extrapolation" and marker.get("storage", {}).get("msg") != "sharded":
        raise ValueError("Extrapolation requires verified causal msg shards.")
    for split in ("train", "msg"):
        if marker.get("storage", {}).get(split) == "sharded":
            manifest_path = Path(processed_dir) / f"{split}_manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"Missing {split} manifest; rerun preprocessing")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("protocol") != PROTOCOL or manifest.get("setting") != setting:
                raise ValueError(f"Incompatible {split} manifest; rerun preprocessing")
            if setting == "extrapolation" and split == "msg":
                if not manifest.get("causal_evaluation"):
                    raise ValueError("Unverified causal msg manifest; rerun preprocessing")
                for shard in manifest["shards"]:
                    assert_causal_support(shard.get("support_time_max"), shard["query_time_min"])
    return marker


def select_context(source_rows, query_min, query_max, setting, context_times,
                   max_edges, evaluation=False):
    if setting not in ("interpolation", "extrapolation"):
        raise ValueError(f"Unknown setting: {setting}")
    if context_times <= 0 or max_edges <= 0:
        raise ValueError("context_times and max_edges must be positive")
    source_times = sorted({row[3] for row in source_rows})
    if setting == "extrapolation":
        # One graph serves a group of queries: the earliest query is the cutoff.
        # Training semantics are preserved; the strict causal condition is for evaluation.
        cutoff = query_min if evaluation else query_max + 1
        candidates = [timestamp for timestamp in source_times if timestamp < cutoff]
        selected_times = set(candidates[-context_times:])
    else:
        ranked_times = sorted(source_times, key=lambda timestamp: (
            0 if query_min <= timestamp <= query_max
            else min(abs(timestamp - query_min), abs(timestamp - query_max)), timestamp))
        selected_times = set(ranked_times[:context_times])
    selected = [row for row in source_rows if row[3] in selected_times]
    if len(selected) > max_edges:
        center = (query_min + query_max) / 2
        selected.sort(key=lambda row: (abs(row[3] - center), row[3], row[0], row[1], row[2]))
        selected = selected[:max_edges]
    return selected


def iter_contexts(source_rows, query_times, setting, shard_time_count,
                  context_times, max_edges, evaluation=False):
    if shard_time_count <= 0:
        raise ValueError("shard_time_count must be positive")
    times = sorted(set(query_times))
    for start in range(0, len(times), shard_time_count):
        group = times[start:start + shard_time_count]
        selected = select_context(source_rows, group[0], group[-1], setting,
                                  context_times, max_edges, evaluation=evaluation)
        if not selected:
            raise ValueError(f"No support facts available for query times {group[0]}–{group[-1]}")
        yield group, selected


def assert_causal_support(support_time_max, query_time):
    if support_time_max is None or int(support_time_max) >= int(query_time):
        raise ValueError(
            f"Non-causal evaluation support: max support time={support_time_max}, "
            f"query time={query_time}; rebuild causal msg shards."
        )
