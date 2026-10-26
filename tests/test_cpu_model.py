"""Exercise the release model with real torch and a CPU sum propagation adapter."""

import ast
import copy
from collections.abc import Sequence
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pine.full_partitions import write_full_partitions
from pine.message import relation_then_time_rotation
from pine.temporal import relative_intervals

ROOT = Path(__file__).resolve().parents[1]


class CPUSumPropagation(nn.Module):
    node_dim = -2

    def propagate(self, edge_index, size, **kwargs):
        message = self.message(kwargs["input"].index_select(self.node_dim, edge_index[0]),
                               kwargs["relation"], kwargs["boundary"], kwargs["edge_type"],
                               kwargs["time_diff"])
        aggregate = self.aggregate(message, kwargs["edge_weight"], edge_index[1], size[1])
        return self.update(aggregate, kwargs["input"])


def cpu_scatter(value, index, dim, dim_size, reduce):
    assert reduce == "sum"
    shape = list(value.shape)
    shape[dim] = dim_size
    return value.new_zeros(shape).index_add_(dim, index, value)


def execute_classes(relative_path, names, scope):
    nodes = [node for node in ast.parse((ROOT / relative_path).read_text()).body
             if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative_path, "exec"), scope)


@pytest.fixture
def reference():
    scope = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
             "copy": copy, "Sequence": Sequence, "MessagePassing": CPUSumPropagation,
             "scatter": cpu_scatter}
    execute_classes("pine/layers.py", {"GeneralizedRelationalConv"}, scope)
    scope["layers"] = types.SimpleNamespace(GeneralizedRelationalConv=scope["GeneralizedRelationalConv"])
    execute_classes("pine/base_nbfnet.py", {"BaseNBFNet"}, scope)
    scope["relation_then_time_rotation"] = relation_then_time_rotation
    execute_classes("pine/layers.py", {"TemporalRelationalConv"}, scope)
    scope["layers"].TemporalRelationalConv = scope["TemporalRelationalConv"]
    from pine.streaming import stream_entity_bellmanford, stream_relation_bellmanford
    scope.update(relative_intervals=relative_intervals,
                 stream_entity_bellmanford=stream_entity_bellmanford,
                 stream_relation_bellmanford=stream_relation_bellmanford)
    execute_classes("pine/models.py", {"PINE", "EntityNBFNet", "RelNBFNet"}, scope)
    return scope["PINE"]


def test_full_model_uses_query_time_and_backpropagates(reference):
    rel = {"class": "RelNBFNet", "input_dim": 4, "hidden_dims": [4, 4],
           "time_range": 10, "message_func": "rotate", "aggregate_func": "sum"}
    ent = {"class": "EntityNBFNet", "input_dim": 4, "hidden_dims": [4, 4],
           "message_func": "rotate", "aggregate_func": "sum"}
    torch.manual_seed(1024)
    model = reference(rel, ent, query_time_range=10).eval()
    graph = types.SimpleNamespace(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_type=torch.tensor([0, 1, 0, 1]), time=torch.tensor([1, 1, 2, 2]),
        num_nodes=3, num_relations=2, num_edges=4,
        relation_graph=types.SimpleNamespace(
            edge_index=torch.tensor([[0, 1], [1, 0]]), edge_type=torch.tensor([0, 1]),
            edge_attr=torch.tensor([1., -1.]), num_nodes=2, num_edges=2))
    batch = torch.tensor([[[0, 0, 0, 3], [0, 1, 0, 3], [0, 2, 0, 3]],
                          [[0, 0, 0, 4], [0, 1, 0, 4], [0, 2, 0, 4]]])
    scores = model(graph, batch)
    assert not torch.allclose(scores[0], scores[1])
    loss = model.compute_loss(scores)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [layer.emb_Time.weight.grad for layer in model.entity_model.layers]
    assert all(gradient is not None for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0
    model.entity_model.use_query_time = False
    off_scores = model(graph, batch)
    assert torch.equal(off_scores[0], off_scores[1])
