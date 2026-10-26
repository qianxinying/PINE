import math

import pytest
import torch

pytest.importorskip("torch_scatter")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data
from pine.models import EntityNBFNet, PINE


def test_joint_model_constructs_and_backpropagates_with_plain_dict_configs():
    rel_cfg = {"class": "RelNBFNet", "input_dim": 4, "hidden_dims": [4],
               "time_range": 10, "message_func": "rotate", "aggregate_func": "sum"}
    ent_cfg = {"class": "EntityNBFNet", "input_dim": 4, "hidden_dims": [4],
               "message_func": "rotate", "aggregate_func": "sum"}
    model = PINE(rel_cfg, ent_cfg, query_time_range=10)
    graph = Data(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]),
        edge_type=torch.tensor([0, 0, 1, 1]),
        time=torch.tensor([2, 3, 2, 3]),
        time_rel=torch.tensor([2, 3, 13, 14]),
        num_nodes=3, num_relations=2,
        relation_graph=Data(edge_index=torch.tensor([[0, 1], [1, 0]]),
                            edge_type=torch.tensor([0, 1]),
                            edge_attr=torch.tensor([1.0, -1.0]), num_nodes=2),
    )
    batch = torch.tensor([[[0, 2, 0, 5], [0, 1, 0, 5]]])
    scores = model(graph, batch)
    loss = model.compute_loss(scores)
    loss.backward()
    assert scores.shape == (1, 2)
    assert torch.isfinite(loss)
    assert model.entity_model.layers[0].emb_Time.weight.grad is not None
    assert rel_cfg["class"] == "RelNBFNet"
    assert ent_cfg["class"] == "EntityNBFNet"


def _make_deterministic_model():
    model = EntityNBFNet(
        input_dim=4,
        hidden_dims=[4],
        query_time_range=10,
        message_func="rotate",
        aggregate_func="sum",
        short_cut=False,
        layer_norm=False,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=1,
    )
    layer = model.layers[0]
    with torch.no_grad():
        first, _, second = layer.relation_projection
        first.weight.copy_(torch.eye(4))
        first.bias.zero_()
        second.weight.copy_(torch.eye(4))
        second.bias.zero_()

        layer.emb_Time.weight.zero_()
        # Edge time is 2, so query times 5 and 6 select deltas 3 and 4.
        layer.emb_Time.weight[10 + 3].zero_()
        layer.emb_Time.weight[10 + 4].fill_(math.pi / 2)

        layer.linear.weight.zero_()
        layer.linear.bias.zero_()
        layer.linear.weight[:, 4:].copy_(torch.eye(4))

        model.mlp[0].weight.zero_()
        model.mlp[0].bias.zero_()
        model.mlp[0].weight[0, 0] = 1.0
    return model


def test_same_support_and_query_relation_change_score_with_query_time():
    model = _make_deterministic_model().eval()
    graph = Data(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_type=torch.tensor([0], dtype=torch.long),
        time=torch.tensor([2], dtype=torch.long),
        num_nodes=2,
        num_relations=2,
    )
    relation_representations = torch.zeros(2, 2, 4)
    relation_representations[..., :2] = 1
    batch = torch.tensor(
        [
            [[0, 1, 0, 5]],
            [[0, 1, 0, 6]],
        ],
        dtype=torch.long,
    )

    scores = model(graph, relation_representations, batch)

    assert scores.shape == (2, 1)
    assert not torch.allclose(scores[0], scores[1])


def test_switch_off_removes_query_time_dependence_without_changing_parameters():
    model = _make_deterministic_model().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    graph = Data(edge_index=torch.tensor([[0], [1]]), edge_type=torch.tensor([0]),
                 time=torch.tensor([2]), num_nodes=2, num_relations=2)
    relations = torch.zeros(2, 2, 4)
    relations[..., :2] = 1
    batch = torch.tensor([[[0, 1, 0, 5]], [[0, 1, 0, 6]]])
    enabled = model(graph, relations, batch)
    model.use_query_time = False
    disabled = model(graph, relations, batch)
    assert not torch.allclose(enabled[0], enabled[1])
    assert torch.equal(disabled[0], disabled[1])
    assert parameter_count == sum(parameter.numel() for parameter in model.parameters())


def test_on_off_construction_has_identical_initial_weights_and_counts():
    rel = {"class": "RelNBFNet", "input_dim": 4, "hidden_dims": [4],
           "time_range": 10, "message_func": "rotate"}
    ent = {"class": "EntityNBFNet", "input_dim": 4, "hidden_dims": [4],
           "message_func": "rotate"}
    torch.manual_seed(1024)
    enabled = PINE(rel, ent, query_time_range=10, use_query_time=True)
    torch.manual_seed(1024)
    disabled = PINE(rel, ent, query_time_range=10, use_query_time=False)
    assert sum(p.numel() for p in enabled.parameters()) == sum(p.numel() for p in disabled.parameters())
    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    for key, value in enabled.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key])


def test_query_relative_interval_is_edgewise_and_signed():
    model = _make_deterministic_model().eval()
    graph = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_type=torch.tensor([0, 1], dtype=torch.long),
        time=torch.tensor([2, 7], dtype=torch.long),
        num_nodes=2,
        num_relations=2,
    )

    delta = model.query_intervals(graph, torch.tensor([5, 6]))

    assert torch.equal(delta, torch.tensor([[3, -2], [4, -1]]))
