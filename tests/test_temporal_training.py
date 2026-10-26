"""Exercise training helpers with real torch but without the PyG extensions."""

import ast
import copy
from functools import reduce
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_definition(relative_path, name, scope):
    path = PROJECT_ROOT / relative_path
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_query_relative_joint_constructor_accepts_dicts_without_mutating_them():
    class Encoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.options = kwargs

    scope = {"torch": torch, "nn": nn, "copy": copy,
             "RelNBFNet": Encoder, "EntityNBFNet": Encoder,
             "EntityNBFNet": Encoder}
    model_class = load_definition("pine/models.py", "PINE", scope)
    rel_cfg = {"class": "RelNBFNet", "input_dim": 4, "hidden_dims": [4]}
    ent_cfg = {"class": "EntityNBFNet", "input_dim": 4, "hidden_dims": [4]}
    model = model_class(rel_cfg, ent_cfg, query_time_range=10)
    assert model.hidden_dim == 4
    assert model.entity_model.options["query_time_range"] == 10
    assert rel_cfg["class"] == "RelNBFNet"
    assert ent_cfg["class"] == "EntityNBFNet"


@pytest.fixture
def edge_match():
    return load_definition("pine/tasks.py", "edge_match", {"torch": torch, "reduce": reduce})


def test_future_query_time_does_not_match_another_relation(edge_match):
    # With bases derived only from support, time=1 carried into relation=1.
    edges = torch.tensor([[0], [1], [0]])
    queries = torch.tensor([[0, 0], [0, 1], [1, 0]])
    indices, counts = edge_match(edges, queries)
    assert indices.tolist() == [0]
    assert counts.tolist() == [0, 1]


def test_missing_query_entity_does_not_match_another_head(edge_match):
    edges = torch.tensor([[1], [0], [0]])
    queries = torch.tensor([[0, 1], [1, 0], [0, 0]])
    indices, counts = edge_match(edges, queries)
    assert indices.tolist() == [0]
    assert counts.tolist() == [0, 1]


def test_edge_match_preserves_duplicate_matches(edge_match):
    edges = torch.tensor([[0, 0, 1], [1, 1, 0], [2, 2, 3]])
    queries = torch.tensor([[0, 1], [1, 0], [2, 3]])
    indices, counts = edge_match(edges, queries)
    assert indices.tolist() == [0, 1, 2]
    assert counts.tolist() == [2, 1]


def test_empty_support_has_no_matches(edge_match):
    indices, counts = edge_match(torch.empty((3, 0), dtype=torch.long), torch.zeros((3, 2), dtype=torch.long))
    assert indices.numel() == 0
    assert counts.tolist() == [0, 0]
