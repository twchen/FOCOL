from collections import defaultdict
import numpy as np
import torch as th
from torch import nn
import dgl
import dgl.ops as F


def count_cooccurs(train_sessions, max_dist):
    counter = defaultdict(
        lambda: defaultdict(lambda: np.zeros(2 * max_dist, dtype=int))
    )
    # counter[i][j]: the weight vector. i <= j
    for sess in train_sessions:
        sess_len = len(sess)
        for i in range(sess_len):
            for j in range(i + 1, sess_len):
                item_i = sess[i]
                item_j = sess[j]
                d = min(max_dist, j - i)
                if item_i < item_j:
                    counter[item_i][item_j][d - 1] += 1
                elif item_j < item_i:
                    counter[item_j][item_i][-d] += 1
                else:
                    counter[item_i][item_j][d - 1] += 1
                    counter[item_j][item_i][-d] += 1
    return counter


def build_graph(train_sessions, num_items, max_dist):
    print('building graph')
    counter = count_cooccurs(train_sessions, max_dist)
    tuples = []
    for item_i in counter:
        for item_j in counter[item_i]:
            tuples.append((item_i, item_j, counter[item_i][item_j]))
            if item_i != item_j:
                tuples.append((item_j, item_i, np.copy(counter[item_i][item_j][::-1])))
    src, dst, cnt = zip(*tuples)
    src = th.tensor(src, dtype=th.long) - 1  # -1 because the item IDs start from 1
    dst = th.tensor(dst, dtype=th.long) - 1
    cnt = th.tensor(cnt, dtype=th.float32)
    graph = dgl.graph((src, dst), num_nodes=num_items)
    graph.edata['cnt'] = cnt
    return graph


class FOGCNConv(nn.Module):
    def __init__(self, num_counts, num_feats):
        super().__init__()
        self.importance = nn.Parameter(
            th.ones(num_counts, num_feats), requires_grad=True
        )

    def extra_repr(self) -> str:
        size = self.importance.size()
        return f'(importance): Parameter({size[0]}, {size[1]})'

    def forward(self, graph, embedding):
        weight = th.softmax(self.importance, dim=0)
        edge_score = graph.edata['cnt'] @ weight  # (num_edges, num_feats)
        new_embedding = F.u_mul_e_sum(
            graph, embedding, edge_score
        )  # (num_nodes, num_feats)
        node_score = F.copy_e_sum(graph, edge_score)  # (num_nodes, num_feats)
        new_embedding = new_embedding / node_score  # normalize
        return new_embedding


class FOGCN(nn.Module):
    def __init__(self, graph, item_feats, max_dist, num_gnn_layers):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleList([
            FOGCNConv(2 * max_dist, item_feats) for _ in range(num_gnn_layers)
        ])

    def forward(self, embedding):
        embeddings = [embedding]
        for layer in self.layers:
            embedding = layer(self.graph, embeddings[-1])
            embeddings.append(embedding)
        final_embedding = th.stack(embeddings, dim=0).mean(0)
        return final_embedding
