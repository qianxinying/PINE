"""Validate either historical context caches or complete msg partitions."""

import json
from pathlib import Path

from .data import protocol

FULL_FORMAT = "pine_full_msg_partitions_v1"


def _safe_file(root, filename):
    path = (root / filename).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Missing or invalid graph partition: {filename}")
    return path


def validate_processed(data_dir, processed_dir, setting):
    root = Path(processed_dir)
    marker_path = root / "storage_format.json"
    if not marker_path.is_file():
        return protocol.validate_processed(data_dir, processed_dir, setting)
    marker = json.loads(marker_path.read_text())
    if marker.get("storage", {}).get("msg") != "full_partitioned":
        return protocol.validate_processed(data_dir, processed_dir, setting)
    if marker.get("protocol") != protocol.PROTOCOL or marker.get("setting") != setting:
        raise ValueError("Full-msg cache protocol/setting does not match this experiment")
    if marker.get("source_sha256") != protocol.source_fingerprints(data_dir):
        raise ValueError("Full-msg partitions do not match the current split/mapping files")
    train = json.loads((root / "train_manifest.json").read_text())
    if train.get("protocol") != protocol.PROTOCOL or train.get("setting") != setting:
        raise ValueError("Incompatible copied train manifest")
    for shard in train["shards"]:
        _safe_file(root, shard["file"])
    msg = json.loads((root / "msg_manifest.json").read_text())
    if (msg.get("format") != FULL_FORMAT or msg.get("protocol") != protocol.PROTOCOL
            or msg.get("setting") != setting):
        raise ValueError("Incompatible full-msg partition manifest")
    if msg.get("causal_evaluation") != (setting == "extrapolation"):
        raise ValueError("Full-msg cache has the wrong causal evaluation policy")
    source_times = msg["source_times"]
    if source_times != sorted(source_times) or not source_times:
        raise ValueError("Invalid full-msg source timestamp index")
    if len(source_times) != msg["source_fact_count"]:
        raise ValueError("Full-msg source fact count mismatch")
    if sum(part["facts"] for part in msg["entity_parts"]) != msg["source_fact_count"]:
        raise ValueError("Full-msg entity partitions do not cover every source fact")
    actual = protocol.read_rows(Path(data_dir) / "msg.txt")
    if sorted(row[3] for row in actual) != source_times:
        raise ValueError("Full-msg timestamp index differs from the source facts")
    for key in ("entity_parts", "relation_parts"):
        parts = msg[key]
        if not parts or len({part["file"] for part in parts}) != len(parts):
            raise ValueError(f"Empty or repeated full-msg {key}")
        previous_max = None
        for part in parts:
            _safe_file(root, part["file"])
            if part["min_time"] > part["max_time"] or (previous_max is not None and part["min_time"] < previous_max):
                raise ValueError("Full-msg partitions are not ordered by activation time")
            previous_max = part["max_time"]
    counts = msg["relation_time_counts"]
    if counts != sorted(counts) or sum(count for _, count in counts) != sum(part["edges"] for part in msg["relation_parts"]):
        raise ValueError("Full-msg relation activation counts do not match the partitions")
    splits = protocol.read_splits(data_dir)
    expected_queries = sorted({row[3] for split in ("valid", "test") for row in splits[split]})
    if msg["query_times"] != expected_queries:
        raise ValueError("Full-msg manifest does not cover all evaluation query times")
    if setting == "extrapolation" and min(source_times) >= min(expected_queries):
        raise ValueError("Some evaluation queries have no strictly past msg facts")
    return marker
