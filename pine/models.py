import copy
from .base_nbfnet import index_to_mask
from .temporal import relative_intervals
from .streaming import stream_entity_bellmanford, stream_relation_bellmanford
import torch
from . import tasks, layers
from .base_nbfnet import BaseNBFNet
from torch import nn, autograd
from torch_scatter import scatter_add

class PINE(nn.Module):
    """Temporal graph completion with relation and query-conditioned entity encoders."""

    def __init__(self, rel_model_cfg, entity_model_cfg, query_time_range=None,
                 use_query_time=True):
        super(PINE, self).__init__()

        rel_model_cfg = copy.deepcopy(dict(rel_model_cfg))
        entity_model_cfg = copy.deepcopy(dict(entity_model_cfg))
        if rel_model_cfg.pop("class", "RelNBFNet") != "RelNBFNet":
            raise ValueError("PINE requires RelNBFNet as its relation encoder")
        if entity_model_cfg.pop("class", "EntityNBFNet") != "EntityNBFNet":
            raise ValueError("PINE requires EntityNBFNet as its entity encoder")
        query_time_range = (rel_model_cfg.get("time_range", 365)
                            if query_time_range is None else query_time_range)
        self.relation_model = RelNBFNet(**rel_model_cfg)
        entity_model_cfg.update(query_time_range=query_time_range, use_query_time=use_query_time)
        self.entity_model = EntityNBFNet(**entity_model_cfg)
        self.hidden_dim = rel_model_cfg["input_dim"]
        self.roletype_embed = nn.Embedding(4, self.hidden_dim)
        relencoder = []
        relencoder.append(nn.Linear(self.hidden_dim + self.hidden_dim * 1, self.hidden_dim, bias=False))
        relencoder.append(nn.ReLU())
        relencoder.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
        relencoder.append(nn.LayerNorm(self.hidden_dim, elementwise_affine=False))
        self.relencoder = nn.Sequential(*relencoder)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / 0.7).log(), requires_grad=True)

    def forward(self, data, batch):

        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels,time_diff=data.relation_graph.edge_attr)
        batch_size = len(batch)
        n_rel = data.num_relations//2
        rel_role_list = torch.zeros(
            batch_size, data.num_relations, dtype=torch.long, device=query_rels.device
        )
        batch_index = torch.arange(batch_size, device=query_rels.device)
        rel_role_list[batch_index, query_rels] = 1
        q_rel_reverse = torch.where(query_rels < n_rel, query_rels+n_rel, query_rels-n_rel)
        rel_role_list[batch_index, q_rel_reverse] = 2
        rel_role_embed = self.roletype_embed(rel_role_list)
        relation_representations = self.relencoder(torch.cat([relation_representations, rel_role_embed], dim=-1))

        score = self.entity_model(data, relation_representations, batch,time_diff=None)
        return score

    def compute_loss(self, pred):
        add_margin = 0.02
        temperature = self.log_inv_t.exp()
        logits = pred / temperature
        if self.training:
            logits[:, 0] -= add_margin
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        exp_logits = torch.exp(logits)
        pos_exp = exp_logits[:, 0]
        neg_exp_sum = exp_logits[:, 1:].sum(dim=-1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum))
        loss = loss.mean()
        return loss

    def visualize(self, data, batch,inverse=False):
        query_rels = batch[0, 2].unsqueeze(0)
        query_time =  batch[0, 3].unsqueeze(0)
        relation_representations = self.relation_model(data.relation_graph, query=query_rels,time_diff=data.relation_graph.edge_attr)
        n_rel = data.num_relations//2
        batch_size = len(batch)
        rel_role_list = torch.zeros(
            batch_size, data.num_relations, dtype=torch.long, device=query_rels.device
        )
        batch_index = torch.arange(batch_size, device=query_rels.device)
        rel_role_list[batch_index, query_rels] = 1
        q_rel_reverse = torch.where(query_rels < n_rel, query_rels+n_rel, query_rels-n_rel)
        rel_role_list[batch_index, q_rel_reverse] = 2
        rel_role_embed = self.roletype_embed(rel_role_list)
        relation_representations = self.relencoder(torch.cat([relation_representations, rel_role_embed], dim=-1))

        path,weight = self.entity_model.visualize(data, relation_representations, batch,time_diff=query_time.unsqueeze(1),inverse=inverse)
        return path,weight


class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        time_range = kwargs.pop("time_range", 365)
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    input_dim=self.dims[i], output_dim=self.dims[i + 1],
                    num_relation=num_relation, time_range=time_range,
                    query_input_dim=self.dims[0], message_func=self.message_func,
                    aggregate_func=self.aggregate_func, layer_norm=self.layer_norm,
                    activation=self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )


    def bellmanford(self, data, h_index,time_diff, separate_grad=False):
        if getattr(data, "is_full_partitioned", False):
            if separate_grad:
                raise ValueError("Full-partition path visualization is not supported")
            return stream_relation_bellmanford(self, data, h_index)

        batch_size = len(h_index)
        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary


        for layer in self.layers:
            hidden = layer(input=layer_input, query=query, boundary=boundary, edge_index=data.edge_index, edge_type=data.edge_type, size=size,edge_weight= edge_weight,time_diff=time_diff)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query,time_diff):
        output = self.bellmanford(rel_graph, h_index=query ,time_diff=time_diff)["node_feature"]
        return output


class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1,
                 query_time_range=365, use_query_time=True, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.query_time_range = int(query_time_range)
        self.use_query_time = bool(use_query_time)
        if self.query_time_range < 1:
            raise ValueError("query_time_range must be a positive integer")
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.TemporalRelationalConv(
                    input_dim=self.dims[i], output_dim=self.dims[i + 1],
                    num_relation=num_relation, time_range=self.query_time_range,
                    query_input_dim=self.dims[0], message_func=self.message_func,
                    aggregate_func=self.aggregate_func, layer_norm=self.layer_norm,
                    activation=self.activation, dependent=False,
                    project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)


    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        """Remove target edges while keeping every edge attribute aligned.

        Temporal intervals and support edges share the same edge mask.
        """
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat(
            [r_index, r_index + data.num_relations // 2], dim=-1
        )
        if self.remove_one_hop:
            edge_index = data.edge_index
            easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
        else:
            edge_index = torch.cat(
                [data.edge_index, data.edge_type.unsqueeze(0)], dim=0
            )
            easy_edge = torch.stack(
                [h_index_ext, t_index_ext, r_index_ext]
            ).flatten(1)

        index = tasks.edge_match(edge_index, easy_edge)[0]
        mask = ~index_to_mask(index, data.num_edges)
        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        data.time = data.time[mask]
        if hasattr(data, "time_rel") and data.time_rel is not None:
            data.time_rel = data.time_rel[mask]
        return data

    def query_intervals(self, data, query_time):
        """Return delta[e,q] = t_q - t_e for every query and support edge."""
        if not hasattr(data, "time") or data.time is None:
            raise ValueError("support graph must provide an edge-level `time` tensor")
        return relative_intervals(
            query_time,
            data.time,
            time_range=self.query_time_range,
        )

    def bellmanford(self, data, h_index, r_index,time_diff, separate_grad=False):
        if getattr(data, "is_full_partitioned", False):
            if separate_grad:
                raise ValueError("Full-partition path visualization is not supported")
            return stream_entity_bellmanford(self, data, h_index, r_index, time_diff)
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(input=layer_input, query=query, boundary=boundary, edge_index=data.edge_index, edge_type=data.edge_type, size=size,edge_weight= edge_weight,time_diff=time_diff)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch, time_diff=None):
        h_index, t_index, r_index, query_time = batch.unbind(-1)
        self.query = relation_representations
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            if getattr(data, "is_full_partitioned", False):
                raise ValueError("Full msg partitions are evaluation support; use train shards for training")
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index,
            t_index,
            r_index,
            num_direct_rel=data.num_relations // 2,
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        assert (query_time[:, [0]] == query_time).all()

        # All candidates belonging to one query share t_q.  The resulting
        # [batch, num_edges] tensor is broadcast through every entity layer.
        if getattr(data, "is_full_partitioned", False):
            # Compute edge-to-query intervals inside each streamed edge part.
            delta_epsilon_q = query_time[:, 0] if self.use_query_time else None
        else:
            delta_epsilon_q = (
                self.query_intervals(data, query_time[:, 0]) if self.use_query_time else None
            )
        output = self.bellmanford(
            data,
            h_index[:, 0],
            r_index[:, 0],
            time_diff=delta_epsilon_q,
        )
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, data, relation_representations,batch,time_diff,inverse=False):
        assert batch.shape == (1, 4)
        time_diff = self.query_intervals(data, batch[:, 3]) if self.use_query_time else None
        h_index, t_index, r_index,t = batch.unbind(-1)
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations


        shape = h_index.shape
        if inverse is True:
            output = self.bellmanford(data, t_index, r_index,time_diff, separate_grad=True)  # (num_nodes, batch_size, feature_dim)
            feature = output["node_feature"]
            index = h_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            # extract representations of tail entities from the updated node states
            feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)
            # probability logit for each tail node in the batch
            # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
            score = self.mlp(feature).squeeze(-1)
            edge_weights = output["edge_weights"]

            edge_grads = autograd.grad(score, edge_weights)
            distances, back_edges = self.beam_search_distance(data, edge_grads, h_index, t_index, self.num_beam)
            paths, weights = self.topk_average_length(distances, back_edges, h_index, self.path_topk)

        else:
            output = self.bellmanford(data, h_index, r_index,time_diff, separate_grad=True)  # (num_nodes, batch_size, feature_dim)
            feature = output["node_feature"]
            index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            # extract representations of tail entities from the updated node states
            feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)
            # probability logit for each tail node in the batch
            # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
            score = self.mlp(feature).squeeze(-1)
            edge_weights = output["edge_weights"]

            edge_grads = autograd.grad(score, edge_weights)
            distances, back_edges = self.beam_search_distance(data, edge_grads, h_index, t_index, self.num_beam)
            paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        # beam search the top-k distance from h to t (and to every other node)
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index

        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            # we don't allow any path goes out of t once it arrives at t
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            time = data.time[edge_mask]
            edge_grad = edge_grad[edge_mask]

            message = input[node_in] + edge_grad.unsqueeze(-1) # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = torch.stack([node_in, node_out, relation,time], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten() # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2) # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                # take the topk messages from the neighborhood
                # distance: (len(node_out_set) * num_beam)
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                # store msg_source for backtracking
                back_edge = msg_source[abs_index] # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 5)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam, 4)
            else:
                distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 5, dtype=torch.long, device=message.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        # backtrack distances and back_edges to generate the paths
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, time,prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r, time)]
                for j in range(i - 1, -1, -1):
                    h, t, r,time, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r, time))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index
