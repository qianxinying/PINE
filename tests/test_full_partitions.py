"""Full-graph coverage, cross-partition relations and layer aggregation tests."""

import ast
import copy
import json
from pathlib import Path
import types

import pytest
import torch
from torch import nn

from pine.data.protocol import PROTOCOL, source_fingerprints
from pine.cache_validation import validate_processed
from pine.full_partitions import FullMessageStore, write_full_partitions
from pine.message import relation_then_time_rotation
from pine.preprocess_full_msg import build_full_msg
from pine.streaming import stream_entity_bellmanford, stream_layers

ROOT = Path(__file__).resolve().parents[1]
ROWS = [(0, 0, 1, 1), (1, 1, 2, 2), (0, 0, 2, 9)]
METADATA = {"num_nodes": 3, "num_relations": 2, "num_times": 10}


def build(tmp_path, setting="interpolation", chunk_size=1):
    write_full_partitions(ROWS, tmp_path, METADATA, setting, [2, 3, 10],
                          PROTOCOL, chunk_size, 3)
    return FullMessageStore(tmp_path, device="cpu", cache_bytes=128)


def entity_rows(graph):
    rows = []
    for part in graph.iter_edge_chunks("cpu"):
        rows.extend(zip(part["edge_index"][0].tolist(), part["edge_type"].tolist(),
                        part["edge_index"][1].tolist(), part["time"].tolist()))
    return rows


def relation_rows(graph):
    result = set()
    for part in graph.relation_graph.iter_edge_chunks("cpu"):
        result.update(zip(part["edge_index"][0].tolist(), part["edge_index"][1].tolist(),
                          part["edge_type"].tolist(), part["edge_attr"].long().tolist()))
    return result


def original_relation_rows(rows):
    # Execute the real original builder with only PyG's Data container stubbed.
    tree = ast.parse((ROOT / "pine/data/graph.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_relation_graph")
    scope = {"torch": torch, "Data": lambda **kwargs: types.SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pine/data/graph.py", "exec"), scope)
    directed = rows + [(t, r + 2, h, time) for h, r, t, time in rows]
    graph = types.SimpleNamespace(
        edge_index=torch.tensor([[h, t] for h, r, t, time in directed]).t(),
        edge_type=torch.tensor([r for h, r, t, time in directed]),
        time=torch.tensor([time for h, r, t, time in directed]), num_relations=4)
    graph = scope["build_relation_graph"](graph).relation_graph
    return set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist(),
                   graph.edge_type.tolist(), graph.edge_attr.long().tolist()))


def test_all_entity_facts_and_inverse_edges_are_used_even_with_one_fact_per_part(tmp_path):
    store = build(tmp_path)
    graph = store.get(3)
    expected = ROWS + [(t, r + 2, h, time) for h, r, t, time in ROWS]
    assert sorted(entity_rows(graph)) == sorted(expected)
    assert graph.num_edges == 6
    assert len(store.manifest["entity_parts"]) == 3
    assert store.cached_bytes <= store.cache_bytes


def test_full_relation_graph_matches_original_builder_including_cross_part_pairs(tmp_path):
    graph = build(tmp_path).get(3)
    assert relation_rows(graph) == original_relation_rows(ROWS)
    # Edge 0 -> relation 1 comes from two facts stored in different parts.
    assert (0, 1, 3, 1) in relation_rows(graph)


@pytest.mark.parametrize("timestamp", [2, 3, 9, 10])
def test_causal_prefix_equals_original_graph_built_from_all_strictly_past_facts(tmp_path, timestamp):
    graph = build(tmp_path, "extrapolation").get(timestamp)
    past = [row for row in ROWS if row[3] < timestamp]
    expected = past + [(t, r + 2, h, time) for h, r, t, time in past]
    assert sorted(entity_rows(graph)) == sorted(expected)
    assert relation_rows(graph) == original_relation_rows(past)
    assert all(time < timestamp for _, _, _, time in entity_rows(graph))


def test_duplicate_relation_witness_uses_earliest_activation(tmp_path):
    rows = ROWS + [(3, 0, 4, 7), (4, 1, 5, 8)]
    metadata = {**METADATA, "num_nodes": 6}
    write_full_partitions(rows, tmp_path, metadata, "extrapolation", [3], PROTOCOL, 1, 3)
    graph = FullMessageStore(tmp_path, "cpu").get(3)
    assert (0, 1, 3, 1) in relation_rows(graph)
    assert relation_rows(graph) == original_relation_rows(ROWS[:2])


def test_support_identity_changes_only_when_a_new_past_fact_becomes_available(tmp_path):
    store = build(tmp_path, "extrapolation")
    assert store.graph_path(3) == store.graph_path(9)
    assert store.graph_path(9) != store.graph_path(10)


@pytest.fixture
def full_cache(tmp_path):
    raw, source, output = tmp_path / "raw", tmp_path / "source", tmp_path / "full"
    raw.mkdir(); source.mkdir()
    text = {"train": "0\t0\t2\t0\n", "msg": "0\t0\t1\t1\n1\t1\t2\t2\n0\t0\t2\t9\n",
            "valid": "0\t1\t2\t3\n", "test": "0\t1\t1\t10\n"}
    for split, contents in text.items():
        (raw / f"{split}.txt").write_text(contents)
    (source / "train_shards").mkdir()
    original_train = source / "train_shards/00000.pt"
    original_train.write_bytes(b"original training graph")
    (source / "train_manifest.json").write_text(json.dumps({
        "protocol": PROTOCOL, "setting": "extrapolation", "time_to_file": {"0": "train_shards/00000.pt"},
        "shards": [{"file": "train_shards/00000.pt"}],
    }))
    (source / "msg_manifest.json").write_text(json.dumps({
        "protocol": PROTOCOL, "setting": "extrapolation", "causal_evaluation": True,
        "shards": [{"support_time_max": 1, "query_time_min": 3}],
    }))
    (source / "storage_format.json").write_text(json.dumps({
        "protocol": PROTOCOL, "setting": "extrapolation",
        "storage": {"train": "sharded", "msg": "sharded"}, "source_sha256": source_fingerprints(raw),
    }))
    build_full_msg(raw, source, output, "extrapolation", "fixture", 1, 3)
    return raw, source, output


def test_preprocessing_copies_training_unchanged_and_validates_complete_cache(full_cache):
    raw, source, output = full_cache
    marker = validate_processed(raw, output, "extrapolation")
    assert marker["storage"] == {"train": "sharded", "msg": "full_partitioned"}
    assert (output / "train_shards/00000.pt").read_bytes() == (source / "train_shards/00000.pt").read_bytes()
    assert FullMessageStore(output, "cpu").get(10).num_edges == 6
    with pytest.raises(FileExistsError):
        build_full_msg(raw, source, output, "extrapolation", "fixture")


def test_missing_part_is_rejected_before_gpu_launch(full_cache):
    raw, _, output = full_cache
    (output / "msg_entity_parts/00001.pt").unlink()
    with pytest.raises(ValueError, match="Missing or invalid graph partition"):
        validate_processed(raw, output, "extrapolation")


def test_changed_source_is_rejected(full_cache):
    raw, _, output = full_cache
    (raw / "msg.txt").write_text((raw / "msg.txt").read_text() + "2\t0\t0\t8\n")
    with pytest.raises(ValueError, match="do not match"):
        validate_processed(raw, output, "extrapolation")


def make_layer():
    tree = ast.parse((ROOT / "pine/layers.py").read_text())
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = [node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name in ("message", "update")]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "pine/layers.py", "exec"), scope)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.dependent = False; self.project_relations = True
            self.relation_projection = nn.Identity()
            self.relation = torch.tensor([[[1., 1., 0., 0.]]])
            self.node_dim = -2; self.message_func = "rotate"; self.time_range = 10
            self.emb_Time = nn.Embedding(21, 2)
            nn.init.zeros_(self.emb_Time.weight)
            self.linear = nn.Linear(8, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.zero_(); self.linear.weight[:, 4:] = torch.eye(4)
            self.layer_norm = None; self.activation = None

    Base.message = scope["message"]; Base.update = scope["update"]
    scope.update(GeneralizedRelationalConv=Base, relation_then_time_rotation=relation_then_time_rotation)
    tree = ast.parse((ROOT / "pine/layers.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemporalRelationalConv")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "layers.py", "exec"), scope)
    return scope["TemporalRelationalConv"]()


class Graph:
    level = "entity"
    num_nodes = 3

    def __init__(self, part_size):
        self.part_size = part_size

    def iter_edge_chunks(self, device):
        for start in range(0, 2, self.part_size):
            edge = torch.tensor([[0, 1], [1, 2]])[:, start:start+self.part_size]
            yield {"edge_index": edge.to(device), "edge_type": torch.zeros(edge.shape[1], dtype=torch.long, device=device),
                   "time": torch.tensor([1, 2], device=device)[start:start+self.part_size]}


@pytest.mark.parametrize("enabled", [False, True])
def test_two_layer_cross_partition_path_and_boundary_match_monolithic_graph(enabled):
    model = types.SimpleNamespace(aggregate_func="sum", short_cut=False, concat_hidden=False,
                                  dims=[4, 4, 4], layers=[make_layer(), make_layer()],
                                  query=torch.tensor([[[1., 1., 0., 0.]]]))
    args = (torch.tensor([0]), torch.tensor([0]), torch.tensor([3]) if enabled else None)
    partitioned = stream_entity_bellmanford(model, Graph(1), *args)["node_feature"]
    monolithic = stream_entity_bellmanford(model, Graph(2), *args)["node_feature"]
    assert torch.equal(partitioned, monolithic)
    # A -> B and B -> C are in DIFFERENT files; two layers must reach C.
    assert torch.equal(partitioned[0, 2, :4], torch.tensor([1., 1., 0., 0.]))
    # Boundary is not multiplied by the number of files.
    assert torch.equal(partitioned[0, 0, :4], torch.tensor([1., 1., 0., 0.]))


def test_all_parts_are_aggregated_before_the_layer_update():
    layer = make_layer()
    model = types.SimpleNamespace(aggregate_func="sum", short_cut=False, layers=[layer])
    query = torch.tensor([[1., 1., 0., 0.]])
    boundary = torch.zeros(1, 3, 4); boundary[:, 0] = query
    result = stream_layers(model, Graph(1), query, boundary)[0]
    # One layer cannot propagate A -> B -> C even if B's part is read later.
    assert torch.equal(result[0, 2], torch.zeros(4))


def test_nonzero_time_rotation_is_independent_of_partition_size():
    layers = [make_layer(), make_layer()]
    for layer in layers:
        with torch.no_grad():
            layer.emb_Time.weight.fill_(0.3)
    model = types.SimpleNamespace(aggregate_func="sum", short_cut=True, concat_hidden=False,
                                  dims=[4, 4, 4], layers=layers, query=torch.tensor([[[1., 1., 0., 0.]]]))
    args = (torch.tensor([0]), torch.tensor([0]), torch.tensor([4]))
    partitioned = stream_entity_bellmanford(model, Graph(1), *args)["node_feature"]
    monolithic = stream_entity_bellmanford(model, Graph(2), *args)["node_feature"]
    assert torch.allclose(partitioned, monolithic)


def test_full_pyg_model_matches_combined_graph_when_dependencies_are_available(tmp_path):
    pytest.importorskip("torch_scatter")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from pine.models import PINE

    graph = build(tmp_path).get(3)
    entity = list(graph.iter_edge_chunks("cpu"))
    relation = list(graph.relation_graph.iter_edge_chunks("cpu"))
    combined = Data(edge_index=torch.cat([part["edge_index"] for part in entity], dim=1),
                    edge_type=torch.cat([part["edge_type"] for part in entity]),
                    time=torch.cat([part["time"] for part in entity]), num_nodes=3, num_relations=4,
                    relation_graph=Data(edge_index=torch.cat([part["edge_index"] for part in relation], dim=1),
                                        edge_type=torch.cat([part["edge_type"] for part in relation]),
                                        edge_attr=torch.cat([part["edge_attr"] for part in relation]), num_nodes=4))
    rel = {"class": "RelNBFNet", "input_dim": 4, "hidden_dims": [4, 4], "time_range": 10,
           "message_func": "rotate", "aggregate_func": "sum"}
    ent = {"class": "EntityNBFNet", "input_dim": 4, "hidden_dims": [4, 4],
           "message_func": "rotate", "aggregate_func": "sum"}
    model = PINE(rel, ent, query_time_range=10).eval()
    batch = torch.tensor([[[0, 2, 0, 3]]])
    with torch.no_grad():
        assert torch.allclose(model(graph, batch), model(combined, batch), atol=1e-5, rtol=1e-5)
        model.entity_model.use_query_time = False
        assert torch.allclose(model(graph, batch), model(combined, batch), atol=1e-5, rtol=1e-5)
