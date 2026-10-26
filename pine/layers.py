import torch
from .message import relation_then_time_rotation
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class GeneralizedRelationalConv(MessagePassing):
    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, time_range,query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=False, project_relations=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.dim=input_dim
        if self.dim % 2:
            raise ValueError("input_dim must be even when using temporal rotate messages")
        self.time_range = time_range
        self.emb_Time = torch.nn.Embedding(time_range * 2 + 1, self.dim // 2)
        r = 6 / np.sqrt(self.dim)
        self.emb_Time.weight.data.uniform_(-r, r)
        self.dropout = nn.Dropout(0.5)
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations


        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                self.relation = nn.Embedding(num_relation, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )

    def forward(self, query, input, edge_index,boundary, edge_type, size,time_diff,edge_weight=None):
        edge_index = edge_index.to(torch.int64)

        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)

        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,size=size,
                                edge_type=edge_type,edge_weight=edge_weight,time_diff=time_diff)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

    def message(self, input_j, relation, boundary,edge_type,time_diff):
        relation_j = relation.index_select(self.node_dim, edge_type)
        if time_diff is not None:
            time_diff_id = (time_diff + self.time_range).long()
            if (time_diff_id < 0).any() or (time_diff_id >= self.emb_Time.num_embeddings).any():
                raise ValueError(
                    "time difference exceeds configured time_range=%d" % self.time_range
                )
            time_embedding = self.emb_Time(time_diff_id)
            d_img = torch.sin(time_embedding)
            d_real = torch.cos(time_embedding)
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            h_real = x_j_re*d_real-x_j_im*d_img
            h_img = x_j_re*d_img+x_j_im*d_real
            out_real = h_real + r_j_re
            out_img = h_img + r_j_im
            message = torch.cat([out_real, out_img], dim=-1)

        else:
            if self.message_func == "transe":
                message = input_j + relation_j
            elif self.message_func == "distmult":
                message = input_j * relation_j
            elif self.message_func == "rotate":
                x_j_re, x_j_im = input_j.chunk(2, dim=-1)
                r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
                message_re = x_j_re * r_j_re - x_j_im * r_j_im
                message_im = x_j_re * r_j_im + x_j_im * r_j_re
                message = torch.cat([message_re, message_im], dim=-1)
            else:
                raise ValueError("Unknown message function `%s`" % self.message_func)

        message = torch.cat([message, boundary], dim=self.node_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)])  # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)
        return output


    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class TemporalRelationalConv(GeneralizedRelationalConv):
    def message(self, input_j, relation, boundary, edge_type, time_diff):
        if time_diff is None:
            return super().message(input_j, relation, boundary, edge_type, None)
        if self.message_func != "rotate":
            raise ValueError("query-time rotation requires message_func='rotate'")
        indices = (time_diff + self.time_range).long()
        if (indices < 0).any() or (indices >= self.emb_Time.num_embeddings).any():
            raise ValueError(f"time difference exceeds configured time_range={self.time_range}")
        relation_j = relation.index_select(self.node_dim, edge_type)
        message = relation_then_time_rotation(input_j, relation_j, self.emb_Time(indices))
        return torch.cat([message, boundary], dim=self.node_dim)
