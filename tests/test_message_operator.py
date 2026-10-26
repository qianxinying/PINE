"""Verify the actual message override using real torch, without PyG imports."""

import argparse
import ast
import math
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pine.message import relation_then_time_rotation

ROOT = Path(__file__).resolve().parents[1]


def complex_tensor(value):
    real, imag = value.chunk(2, dim=-1)
    return torch.complex(real, imag)


@pytest.fixture
def conv():
    # Execute the original message method and the new override unchanged.
    # Only PyG's propagation machinery and constructor are excluded here.
    tree = ast.parse((ROOT / "pine/layers.py").read_text())
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == "message")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pine/layers.py", "exec"), scope)

    class Base:
        def __init__(self):
            self.node_dim = -2
            self.message_func = "rotate"
            self.time_range = 10
            self.emb_Time = torch.nn.Embedding(21, 2).double()

    Base.message = scope["message"]
    scope.update(GeneralizedRelationalConv=Base, relation_then_time_rotation=relation_then_time_rotation)
    tree = ast.parse((ROOT / "pine/layers.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemporalRelationalConv")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pine/layers.py", "exec"), scope)
    return scope["TemporalRelationalConv"]()


@pytest.fixture
def inputs():
    source = torch.tensor([[[1., 2., 3., 4.], [2., 1., -1., 3.]]], dtype=torch.double)
    relation = torch.tensor([[[2., 3., 4., -2.], [3., 2., -2., 4.]]], dtype=torch.double)
    boundary = torch.tensor([[[9., 8., 7., 6.]]], dtype=torch.double)
    return source, relation, boundary, torch.tensor([1, 0])


def test_disabled_message_is_exactly_original_message(conv, inputs):
    actual = conv.message(*inputs, None)
    original = super(type(conv), conv).message(*inputs, None)
    assert torch.equal(actual, original)


def test_zero_phase_is_exactly_original_relation_product(conv, inputs):
    with torch.no_grad():
        conv.emb_Time.weight.zero_()
    actual = conv.message(*inputs, torch.tensor([[3, -2]]))
    assert torch.equal(actual, super(type(conv), conv).message(*inputs, None))


def test_nonzero_phase_rotates_relation_product_and_preserves_boundary(conv, inputs):
    phase = torch.tensor([[[0., math.pi / 2], [-math.pi / 2, math.pi]]], dtype=torch.double)
    with torch.no_grad():
        conv.emb_Time.weight[13] = phase[0, 0]
        conv.emb_Time.weight[8] = phase[0, 1]
    source, relation, boundary, edge_type = inputs
    actual = conv.message(*inputs, torch.tensor([[3, -2]]))
    expected = complex_tensor(source) * complex_tensor(relation[:, edge_type]) * torch.exp(1j * phase)
    assert torch.allclose(complex_tensor(actual[:, :2]), expected)
    assert torch.equal(actual[:, 2:], boundary)


def test_zero_relation_produces_zero_message_even_with_time_rotation():
    source = torch.tensor([[2., 3., 4., 5.]])
    result = relation_then_time_rotation(source, torch.zeros_like(source), torch.ones(1, 2))
    assert torch.equal(result, torch.zeros_like(source))


def test_time_source_and_relation_receive_gradients(conv, inputs):
    source, relation, boundary, edge_type = inputs
    source.requires_grad_()
    relation.requires_grad_()
    result = conv.message(source, relation, boundary, edge_type, torch.tensor([[3, -2]]))
    result.square().sum().add(result.sum()).backward()
    for gradient in (source.grad, relation.grad, conv.emb_Time.weight.grad):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_interval_bounds_are_checked(conv, inputs):
    with pytest.raises(ValueError, match="exceeds configured time_range"):
        conv.message(*inputs, torch.tensor([[11, 0]]))


@pytest.mark.parametrize("mode", ["on", "off"])
def test_cli_accepts_query_time_switch(monkeypatch, mode):
    tree = ast.parse((ROOT / "script/train.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    scope = {"argparse": argparse, "PROJECT_ROOT": str(ROOT)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "experiment.py", "exec"), scope)
    monkeypatch.setattr("sys.argv", ["experiment.py", "--dataset", "ICEWS14", "--setting", "interpolation",
                                    "--time-range", "365", "--query-time", mode])
    assert scope["parse_args"]().query_time == mode


def test_collector_finds_both_switch_modes_in_new_result_directories(tmp_path):
    for mode in ("on", "off"):
        directory = tmp_path / "interpolation/ICEWS14" / mode / "PINE/JointDataset/run"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({
            "dataset": "ICEWS14", "setting": "interpolation",
            "query_time_enabled": mode == "on",
            "seed": 1024, "num_epoch": 3, "best_epoch": 2, "best_validation_mrr": 0.3,
            "test": {"mr": 10, "mrr": 0.3, "hits@1": 0.2, "hits@3": 0.3, "hits@10": 0.5},
        }))
    output = tmp_path / "results.csv"
    result = subprocess.run([sys.executable, str(ROOT / "script/collect_results.py"),
                             "--run-root", str(tmp_path), "--output", str(output)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    with output.open() as fin:
        rows = list(csv.DictReader(fin))
    assert len(rows) == 2
    assert {row["query_time_enabled"] for row in rows} == {"True", "False"}
