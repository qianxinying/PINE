"""Complete msg graphs stored and evaluated as edge partitions."""

from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict
import json
import logging
from pathlib import Path
import sqlite3

import torch

FORMAT = "pine_full_msg_partitions_v1"


def relation_witnesses(rows, num_direct_relations):
    """Global relation edges, including witnesses crossing entity partitions.

    An edge becomes available once BOTH supporting events are available.
    SQL reduces duplicate witnesses to their earliest activation timestamp.
    """
    heads, tails = defaultdict(set), defaultdict(set)
    for h, r, t, time in rows:
        heads[h].add((r, time)); tails[t].add((r, time))
        heads[t].add((r + num_direct_relations, time))
        tails[h].add((r + num_direct_relations, time))
    for node in sorted(heads.keys() | tails.keys()):
        head, tail = sorted(heads[node]), sorted(tails[node])
        for left, right, kind in ((head, head, 0), (tail, tail, 1),
                                  (head, tail, 2), (tail, head, 3)):
            for r1, t1 in left:
                for r2, t2 in right:
                    yield r1, r2, kind, t2 - t1, max(t1, t2)


def write_full_partitions(rows, output_dir, metadata, setting, query_times,
                          protocol, entity_chunk_size=5000, relation_chunk_size=4096):
    """Write ALL facts once; limits are partition sizes, never edge caps."""
    if not rows or entity_chunk_size < 1 or relation_chunk_size < 1:
        raise ValueError("Full msg graph must be nonempty and partition sizes positive")
    output = Path(output_dir)
    entity_dir, relation_dir = output / "msg_entity_parts", output / "msg_relation_parts"
    entity_dir.mkdir(parents=True, exist_ok=True)
    relation_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row[3], row[0], row[1], row[2]))
    entity_parts = []
    num_direct = metadata["num_relations"]
    for start in range(0, len(rows), entity_chunk_size):
        subset = rows[start:start + entity_chunk_size]
        direct = torch.tensor([[row[0], row[2]] for row in subset], dtype=torch.long).t()
        edge_type = torch.tensor([row[1] for row in subset], dtype=torch.long)
        time = torch.tensor([row[3] for row in subset], dtype=torch.long)
        # Interleave direct/inverse edges to keep the partitions time sorted.
        graph = {
            "edge_index": torch.stack([direct, direct.flip(0)], dim=2).flatten(1),
            "edge_type": torch.stack([edge_type, edge_type + num_direct], dim=1).flatten(),
            "time": time.repeat_interleave(2),
        }
        filename = f"msg_entity_parts/{len(entity_parts):05d}.pt"
        torch.save(graph, output / filename)
        entity_parts.append({"file": filename, "facts": len(subset),
                             "edges": len(subset) * 2, "min_time": subset[0][3],
                             "max_time": subset[-1][3]})

    # Keep relation-edge deduplication on disk rather than a potentially huge
    # Python set. This reproduces build_relation_graph's global edge union.
    database = output / "relation_build.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute("CREATE TABLE edges (r1 INTEGER, r2 INTEGER, kind INTEGER, "
                           "delta INTEGER, active INTEGER, PRIMARY KEY(r1,r2,kind,delta)) WITHOUT ROWID")
        statement = ("INSERT INTO edges VALUES (?,?,?,?,?) ON CONFLICT(r1,r2,kind,delta) "
                     "DO UPDATE SET active=MIN(active,excluded.active)")
        pending = []
        for witness in relation_witnesses(rows, num_direct):
            pending.append(witness)
            if len(pending) >= 10000:
                connection.executemany(statement, pending)
                pending.clear()
        if pending:
            connection.executemany(statement, pending)
        connection.commit()
        relation_parts, activation_counts = [], Counter()
        cursor = connection.execute("SELECT r1,r2,kind,delta,active FROM edges "
                                    "ORDER BY active,r1,r2,kind,delta")
        while True:
            subset = cursor.fetchmany(relation_chunk_size)
            if not subset:
                break
            values = torch.tensor(subset, dtype=torch.long)
            graph = {"edge_index": values[:, :2].t().contiguous(),
                     "edge_type": values[:, 2].clone(), "edge_attr": values[:, 3].float(),
                     "active_time": values[:, 4].clone()}
            filename = f"msg_relation_parts/{len(relation_parts):05d}.pt"
            torch.save(graph, output / filename)
            relation_parts.append({"file": filename, "edges": len(subset),
                                   "min_time": subset[0][4], "max_time": subset[-1][4]})
            activation_counts.update(row[4] for row in subset)
    database.unlink()
    source_times = [row[3] for row in rows]
    query_times = sorted(set(query_times))
    if setting == "extrapolation" and any(bisect_left(source_times, time) == 0 for time in query_times):
        raise ValueError("Some evaluation queries have no strictly past msg facts")
    manifest = {"format": FORMAT, "protocol": protocol, "setting": setting,
                "causal_evaluation": setting == "extrapolation", "metadata": metadata,
                "source_fact_count": len(rows), "source_times": source_times,
                "relation_time_counts": sorted(activation_counts.items()),
                "query_times": query_times, "entity_parts": entity_parts,
                "relation_parts": relation_parts}
    (output / "msg_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[full msg] facts={len(rows)} entity_parts={len(entity_parts)} "
          f"relation_edges={sum(activation_counts.values())} relation_parts={len(relation_parts)}", flush=True)
    return manifest


class FullGraphView:
    """One complete graph (or complete past prefix), traversed partition by partition."""
    is_full_partitioned = True

    def __init__(self, store, cutoff, level="entity"):
        self.store, self.cutoff, self.level = store, cutoff, level
        metadata = store.manifest["metadata"]
        self.num_nodes = metadata["num_nodes"] if level == "entity" else metadata["num_relations"] * 2
        self.num_relations = metadata["num_relations"] * 2 if level == "entity" else 4
        if level == "entity":
            times = store.manifest["source_times"]
            count = len(times) if cutoff is None else bisect_left(times, cutoff)
            self.num_edges = count * 2
        else:
            self.num_edges = sum(count for time, count in store.manifest["relation_time_counts"]
                                 if cutoff is None or time < cutoff)
        self.edge_attr = None  # Relation time differences are read per partition.
        if level == "entity":
            self.relation_graph = FullGraphView(store, cutoff, "relation")

    def iter_edge_chunks(self, device):
        key = "entity_parts" if self.level == "entity" else "relation_parts"
        time_key = "time" if self.level == "entity" else "active_time"
        for part in self.store.manifest[key]:
            if self.cutoff is not None and part["min_time"] >= self.cutoff:
                break
            graph = self.store.read_part(part["file"])
            mask = None if self.cutoff is None else graph[time_key] < self.cutoff
            if mask is not None and not mask.any():
                continue
            result = {}
            for name, tensor in graph.items():
                value = tensor if mask is None else tensor[:, mask] if name == "edge_index" else tensor[mask]
                result[name] = value.to(device)
            yield result


class FullMessageStore:
    def __init__(self, processed_dir, device="cuda:0", cache_bytes=256 * 1024 * 1024):
        self.root = Path(processed_dir)
        self.manifest = json.loads((self.root / "msg_manifest.json").read_text())
        self.device = device
        self.cache_bytes = cache_bytes
        self.cache, self.cached_bytes = OrderedDict(), 0
        self.current_key, self.current_graph = None, None

    def graph_path(self, timestamp):
        if self.manifest["setting"] == "interpolation":
            return str(self.root / "msg_manifest.json") + "#all"
        count = bisect_left(self.manifest["source_times"], int(timestamp))
        return str(self.root / "msg_manifest.json") + f"#past_facts={count}"

    def get(self, timestamp):
        key = self.graph_path(timestamp)
        if key != self.current_key:
            cutoff = int(timestamp) if self.manifest["setting"] == "extrapolation" else None
            graph = FullGraphView(self, cutoff)
            if graph.num_edges == 0:
                raise ValueError(f"No strictly past msg facts for query time {timestamp}")
            self.current_key, self.current_graph = key, graph
            logging.getLogger().info("Full msg graph: cutoff=%s facts=%d relation_edges=%d",
                                     cutoff, graph.num_edges // 2, graph.relation_graph.num_edges)
        return self.current_graph

    def read_part(self, filename):
        if filename in self.cache:
            value, size = self.cache.pop(filename)
            self.cache[filename] = value, size
            return value
        # Partitions contain only ordinary tensor dictionaries.
        value = torch.load(self.root / filename, map_location="cpu", weights_only=True)
        size = sum(tensor.numel() * tensor.element_size() for tensor in value.values())
        while self.cache and self.cached_bytes + size > self.cache_bytes:
            _, (_, old_size) = self.cache.popitem(last=False)
            self.cached_bytes -= old_size
        if size <= self.cache_bytes:
            self.cache[filename] = value, size
            self.cached_bytes += size
        return value

    def clear(self):
        self.cache.clear()
        self.cached_bytes = 0
        self.current_key, self.current_graph = None, None


def install_full_message_runtime(pretrain):
    """Adapt this experiment's helper module in memory; no shared files change."""
    original_store = pretrain.GraphStore
    original_metadata = pretrain.load_graph_metadata

    def store_factory(processed_dir, split, device="cuda:0"):
        marker = json.loads((Path(processed_dir) / "storage_format.json").read_text())
        if split == "msg" and marker.get("storage", {}).get("msg") == "full_partitioned":
            return FullMessageStore(processed_dir, device)
        return original_store(processed_dir, split, device)

    def metadata_loader(processed_dir, split="msg"):
        store = store_factory(processed_dir, split, "cpu")
        if isinstance(store, FullMessageStore):
            return store.get(min(store.manifest["query_times"]))
        return original_metadata(processed_dir, split)

    pretrain.GraphStore = store_factory
    pretrain.load_graph_metadata = metadata_loader
