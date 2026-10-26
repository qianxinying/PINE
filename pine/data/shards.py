"""Build temporal graph shards."""

import json
import torch
from torch_geometric.data import Data
from .graph import build_relation_graph
from .protocol import PROTOCOL, iter_contexts


def make_graph(rows, num_nodes, num_relations, num_times):
    if not rows:
        raise ValueError("Cannot build an empty graph shard")

    target_edge_index = torch.tensor(
        [[row[0], row[2]] for row in rows], dtype=torch.long
    ).t().contiguous()
    target_edge_type = torch.tensor([row[1] for row in rows], dtype=torch.long)
    target_time = torch.tensor([row[3] for row in rows], dtype=torch.long)

    edge_index = torch.cat([target_edge_index, target_edge_index.flip(0)], dim=1)
    edge_type = torch.cat([target_edge_type, target_edge_type + num_relations])
    time = torch.cat([target_time, target_time])
    time_rel = torch.cat([
        target_edge_type * (num_times + 1) + target_time,
        (target_edge_type + num_relations) * (num_times + 1) + target_time,
    ])

    graph = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        train_time=time,
        target_edge_index=target_edge_index,
        time_rel=time_rel,
        target_edge_type=target_edge_type,
        num_time=num_times,
        num_relations=num_relations * 2,
        time=time,
    )
    return build_relation_graph(graph)


def write_shards(split, source_rows, query_times, output_dir, setting,
                 shard_time_count, context_times, max_edges, metadata):
    shard_dir = output_dir / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "pine_sharded_v1",
        "split": split,
        "setting": setting,
        "protocol": PROTOCOL,
        "causal_evaluation": setting == "extrapolation" and split == "msg",
        "shard_time_count": shard_time_count,
        "context_times": context_times,
        "max_edges": max_edges,
        "time_to_file": {},
        "shards": [],
    }

    contexts = iter_contexts(source_rows, query_times, setting, shard_time_count,
                             context_times, max_edges, evaluation=(split == "msg"))
    for shard_id, (times, rows) in enumerate(contexts):
        if setting == "extrapolation" and split == "msg":
            if any(row[3] >= min(times) for row in rows):
                raise ValueError(
                    f"Non-causal msg support selected for query times "
                    f"{min(times)}-{max(times)}"
                )
        if not rows:
            raise ValueError(
                f"No support edges selected for {split} query times {min(times)}-{max(times)}"
            )
        graph = make_graph(rows, **metadata)
        relative_file = f"{split}_shards/{shard_id:05d}.pt"
        torch.save(graph, output_dir / relative_file)
        for timestamp in times:
            manifest["time_to_file"][str(timestamp)] = relative_file
        manifest["shards"].append({
            "file": relative_file,
            "query_time_min": min(times),
            "query_time_max": max(times),
            "query_time_count": len(times),
            "support_edges": len(rows),
            "support_time_min": min(row[3] for row in rows),
            "support_time_max": max(row[3] for row in rows),
            "directed_edges": graph.num_edges,
            "relation_edges": graph.relation_graph.num_edges,
        })
        print(
            f"[{split}] {relative_file}: queries={min(times)}-{max(times)} "
            f"support={len(rows)} relation_edges={graph.relation_graph.num_edges}"
        )

    manifest_path = output_dir / f"{split}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2)
    print(f"Saved manifest: {manifest_path}")
