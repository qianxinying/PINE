"""Global sum aggregation over ALL edge partitions at every message layer."""

import torch


def stream_layers(model, graph, query, boundary, time_diff=None):
    """Keep global node states; partition only the edge-message workload.

    Applying update/nonlinearity independently to each partition would change
    the model. Here every partition contributes to ONE global sum first.
    """
    if model.aggregate_func != "sum":
        raise ValueError("Full-graph streaming currently requires aggregate_func='sum'")
    layer_input = boundary
    hiddens = []
    for layer in model.layers:
        if layer.dependent:
            relation = layer.relation_linear(query).view(len(query), layer.num_relation, layer.input_dim)
        elif layer.project_relations:
            relation = layer.relation_projection(layer.relation)
        else:
            relation = layer.relation.weight.expand(len(query), -1, -1)
        total = torch.zeros_like(layer_input)
        empty_boundary = boundary[:, :0]
        for edges in graph.iter_edge_chunks(query.device):
            source, target = edges["edge_index"]
            source_features = layer_input.index_select(1, source)
            if graph.level == "relation":
                interval = edges["edge_attr"]
            elif time_diff is None:
                interval = None
            else:
                # time_diff contains the batch's query timestamps, not an
                # all-edges tensor. Each partition supplies its own edge time.
                interval = time_diff[:, None] - edges["time"][None, :]
            messages = layer.message(source_features, relation, empty_boundary,
                                     edges["edge_type"], interval)
            total.index_add_(1, target, messages)
        # PINE's boundary contribution is added ONCE per layer, not per part.
        hidden = layer.update(total + boundary, layer_input)
        if model.short_cut and hidden.shape == layer_input.shape:
            hidden = hidden + layer_input
        hiddens.append(hidden)
        layer_input = hidden
    return hiddens


def stream_entity_bellmanford(model, graph, h_index, r_index, query_time):
    query = model.query[torch.arange(len(r_index), device=r_index.device), r_index]
    boundary = query.new_zeros(len(query), graph.num_nodes, model.dims[0])
    index = h_index.unsqueeze(-1).expand_as(query)
    boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
    hiddens = stream_layers(model, graph, query, boundary, query_time)
    node_query = query[:, None].expand(-1, graph.num_nodes, -1)
    feature = torch.cat((hiddens if model.concat_hidden else hiddens[-1:]) + [node_query], dim=-1)
    return {"node_feature": feature, "edge_weights": []}


def stream_relation_bellmanford(model, graph, h_index):
    query = torch.ones(len(h_index), model.dims[0], device=h_index.device,
                       dtype=model.layers[0].linear.weight.dtype)
    boundary = query.new_zeros(len(query), graph.num_nodes, model.dims[0])
    index = h_index.unsqueeze(-1).expand_as(query)
    boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
    hiddens = stream_layers(model, graph, query, boundary)
    if model.concat_hidden:
        node_query = query[:, None].expand(-1, graph.num_nodes, -1)
        feature = model.mlp(torch.cat(hiddens + [node_query], dim=-1))
    else:
        feature = hiddens[-1]
    return {"node_feature": feature, "edge_weights": []}
