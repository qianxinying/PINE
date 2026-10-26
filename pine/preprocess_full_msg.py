"""Build complete msg partitions; copy the existing training shards unchanged."""

import argparse
import json
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from pine.data import protocol
from pine.full_partitions import write_full_partitions
from pine.cache_validation import validate_processed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setting", choices=["interpolation", "extrapolation"], required=True)
    parser.add_argument("--raw-root", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--source-processed-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--entity-chunk-size", type=int, default=5000)
    parser.add_argument("--relation-chunk-size", type=int, default=4096)
    return parser.parse_args()


def build_full_msg(raw, source, output, setting, dataset, entity_chunk_size=5000, relation_chunk_size=4096):
    raw, source, output = Path(raw), Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    if entity_chunk_size < 1 or relation_chunk_size < 1:
        raise ValueError("Partition sizes must be positive")
    # Read/check inputs before copying any training artifacts.
    splits = protocol.read_splits(raw)
    protocol.validate_splits(splits)
    validate_processed(raw, source, setting)
    train = json.loads((source / "train_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for shard in train["shards"]:
        relative = Path(shard["file"])
        original = (source / relative).resolve()
        if relative.is_absolute() or source.resolve() not in original.parents:
            raise ValueError("Invalid source train shard path")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    shutil.copy2(source / "train_manifest.json", output / "train_manifest.json")
    all_rows = [row for rows in splits.values() for row in rows]
    metadata = {"num_nodes": max(max(row[0], row[2]) for row in all_rows) + 1,
                "num_relations": max(row[1] for row in all_rows) + 1,
                "num_times": max(row[3] for row in all_rows) + 1}
    queries = [row[3] for split in ("valid", "test") for row in splits[split]]
    write_full_partitions(splits["msg"], output, metadata, setting, queries,
                          protocol.PROTOCOL, entity_chunk_size, relation_chunk_size)
    marker = {"format": "pine_storage_v1", "dataset": dataset,
              "protocol": protocol.PROTOCOL, "setting": setting,
              "storage": {"train": "sharded", "msg": "full_partitioned"},
              "source_sha256": protocol.source_fingerprints(raw),
              "support_mode": "full_msg_partitions"}
    (output / "storage_format.json").write_text(json.dumps(marker, indent=2))
    validate_processed(raw, output, setting)
    print(f"[verified full msg] {output}", flush=True)


def main():
    args = parse_args()
    suffix = Path(args.setting) / args.dataset
    build_full_msg(Path(args.raw_root).expanduser().resolve() / suffix,
                   Path(args.source_processed_root).expanduser().resolve() / suffix,
                   Path(args.output_root).expanduser().resolve() / suffix,
                   args.setting, args.dataset, args.entity_chunk_size, args.relation_chunk_size)


if __name__ == "__main__":
    main()
