"""Construct relation graphs with event-to-event time intervals."""

import torch
from torch_geometric.data import Data


def build_relation_graph(graph):
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_rels = graph.num_relations
    device = edge_index.device
    time = graph.time


    hh_edges = set()  # (r1, r2, 0, delta_t)
    tt_edges = set()  # (r1, r2, 1, delta_t)
    ht_edges = set() # (r1, r2, 2, delta_t)
    th_edges = set() # (r1, r2, 3, delta_t)


    head_dict = {}  # {node: [[r1, t1], [r2, t2], ...]}
    tail_dict = {}


    for i in range(edge_index.size(1)):
        h = edge_index[0, i].item()
        t = edge_index[1, i].item()
        r = edge_type[i].item()
        ti = time[i].item()


        if h not in head_dict:
            head_dict[h] = []
        head_dict[h].append( (r, ti) )


        if t not in tail_dict:
            tail_dict[t] = []
        tail_dict[t].append( (r, ti) )


    for node in head_dict:
        relations = head_dict[node]
        n = len(relations)
        for i in range(n):
            r1, t1 = relations[i]
            for j in range(n):
                r2, t2 = relations[j]
                delta = t2 - t1
                hh_edges.add( (r1, r2, 0, delta) )


    for node in tail_dict:
        relations = tail_dict[node]
        n = len(relations)
        for i in range(n):
            r1, t1 = relations[i]
            for j in range(n):
                r2, t2 = relations[j]
                delta = t2 - t1
                tt_edges.add( (r1, r2, 1, delta) )


    for node in set(head_dict.keys()) & set(tail_dict.keys()):
        head_relations = head_dict[node]
        tail_relations = tail_dict[node]
        for (r1, t1) in head_relations:
            for (r2, t2) in tail_relations:
                delta = t2 - t1
                ht_edges.add( (r1, r2, 2, delta) )


    for node in set(tail_dict.keys()) & set(head_dict.keys()):
        tail_relations = tail_dict[node]
        head_relations = head_dict[node]
        for (r1, t1) in tail_relations:
            for (r2, t2) in head_relations:
                delta = t2 - t1
                th_edges.add( (r1, r2, 3, delta) )


    def to_tensor(data):
        return (torch.tensor(data, dtype=torch.long, device=device)
                if data else torch.empty((0, 4), dtype=torch.long, device=device))

    hh_tensor = to_tensor(list(hh_edges))
    tt_tensor = to_tensor(list(tt_edges))
    ht_tensor = to_tensor(list(ht_edges))
    th_tensor = to_tensor(list(th_edges))


    all_edges = torch.cat([hh_tensor, tt_tensor, ht_tensor, th_tensor], dim=0)


    rel_graph = Data(
        edge_index=all_edges[:, :2].t().contiguous(),  # (2, num_edges)
        edge_type=all_edges[:, 2],
        edge_attr=all_edges[:, 3].float(),
        num_nodes=num_rels,
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph
