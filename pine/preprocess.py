"""Choose a memory-safe storage format per graph split.

Training graphs can be very large, so ``train`` is always written as temporal
shards.  Message-graph sharding can be enabled for every dataset with
``--shard-msg``; this is required for causal extrapolation evaluation.
"""

import argparse
import json
from pathlib import Path

import torch

from .data.shards import make_graph, write_shards
from .data.protocol import PROTOCOL, read_splits, source_fingerprints, validate_processed, validate_splits


SHARDED_MSG_DATASETS = {"ICEWS14", "ICEWS05-15"}


def write_storage_marker(output_dir, storage_format, dataset, setting, data_dir):
    marker = {
        "format": "pine_storage_v1",
        "dataset": dataset,
        "storage": storage_format,
        "protocol": PROTOCOL,
        "setting": setting,
        "source_sha256": source_fingerprints(data_dir),
    }
    with (output_dir / "storage_format.json").open("w", encoding="utf-8") as fout:
        json.dump(marker, fout, indent=2)


def process_mixed_dataset(data_dir, output_dir, dataset, setting,
                          shard_time_count, context_times, max_edges,
                          shard_msg=False):
    """Shard train, and shard the message graph for relation-heavy datasets."""
    splits = read_splits(data_dir)
    validate_splits(splits)
    (output_dir / "storage_format.json").unlink(missing_ok=True)
    all_rows = sum(splits.values(), [])
    metadata = {
        "num_nodes": max(max(row[0], row[2]) for row in all_rows) + 1,
        "num_relations": max(row[1] for row in all_rows) + 1,
        "num_times": max(row[3] for row in all_rows) + 1,
    }
    write_shards(
        "train", splits["train"], [row[3] for row in splits["train"]],
        output_dir, setting, shard_time_count, context_times, max_edges,
        metadata,
    )
    if setting == "extrapolation" or shard_msg or dataset in SHARDED_MSG_DATASETS:
        evaluation_times = [row[3] for row in splits["valid"] + splits["test"]]
        write_shards(
            "msg", splits["msg"], evaluation_times,
            output_dir, setting, shard_time_count, context_times, max_edges,
            metadata,
        )
        msg_storage = "sharded"
    else:
        msg_graph = make_graph(splits["msg"], **metadata)
        msg_path = output_dir / "msg.pt"
        torch.save(msg_graph, msg_path)
        print(
            f"[msg] {msg_path}: support={len(splits['msg'])} "
            f"relation_edges={msg_graph.relation_graph.num_edges}"
        )
        msg_storage = "monolithic"
    return msg_storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build memory-bounded train/msg graph storage"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--setting", choices=["interpolation", "extrapolation"], required=True)
    parser.add_argument("--shard-time-count", type=int, default=30)
    parser.add_argument("--context-times", type=int, default=60)
    parser.add_argument("--max-edges", type=int, default=5000)
    parser.add_argument(
        "--shard-msg", action="store_true",
        help="Build query-time message shards for every dataset, including datasets "
             "that normally use a monolithic msg.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    msg_storage = process_mixed_dataset(
        data_dir, output_dir, args.dataset, args.setting, args.shard_time_count,
        args.context_times, args.max_edges, shard_msg=args.shard_msg,
    )
    print(f"[{args.dataset}] train=sharded msg={msg_storage}")
    write_storage_marker(
        output_dir,
        {"train": "sharded", "msg": msg_storage},
        args.dataset, args.setting, data_dir,
    )
    validate_processed(data_dir, output_dir, args.setting)


if __name__ == "__main__":
    main()
