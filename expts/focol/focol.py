from .argparse import make_parser

parser = make_parser()
args = parser.parse_args()

import logging
import torch as th
from torch import nn
from utils.cached_property import cached_property
from expts.expt import ExperimentBase
from .collate import pad_seqs_and_shift_labels, TrainLoaderCollateFn
from .FOGCN import FOGCN, build_graph
from .session_encoder import SessionEncoder


class FOCOL(nn.Module):
    def __init__(
        self, graph, num_items, item_feats, max_len, num_gnn_layers, num_enc_layers,
        **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_items, item_feats)
        self.graph = graph
        if num_gnn_layers > 0:
            self.graph_embedding = FOGCN(graph, item_feats, max_len - 1, num_gnn_layers)
        else:
            self.graph_embedding = None
        self.mask_embedding = nn.Parameter(
            th.randn(1, item_feats, dtype=th.float32), requires_grad=True
        )
        self.sess_encoder = SessionEncoder(
            item_feats,
            num_layers=num_enc_layers,
            max_len=max_len,
        )

    def compute_global_embeddings(self):
        if self.graph_embedding is not None:
            embedding = self.graph_embedding(self.embedding.weight)
        else:
            embedding = self.embedding.weight
        embedding = th.cat([self.mask_embedding, embedding], dim=0)
        return embedding

    def compute_sess_embeddings(self, global_embeddings, inputs, **kwargs):
        sess_embeddings = self.sess_encoder(global_embeddings, *inputs, **kwargs)
        return sess_embeddings

    def forward(self, *inputs):
        global_embeddings = self.compute_global_embeddings()
        sess_embeddings = self.compute_sess_embeddings(global_embeddings, inputs)
        logits = sess_embeddings @ global_embeddings[1:].t()
        return logits


class Experiment(ExperimentBase):
    def __init__(self, args):
        super().__init__(args)
        args.ModelClass = FOCOL

    def read_dataset(self):
        super().read_dataset()
        graph = build_graph(
            train_sessions=self.args.train_sessions,
            num_items=self.args.num_items,
            max_dist=self.args.max_len - 1,
        )
        self.args.graph = graph.to(self.args.device)

    @cached_property
    def train_loader(self):
        train_loader = super().train_loader
        train_loader.collate_fn = TrainLoaderCollateFn(
            **self.args, collate_fn=self.collate_fn
        )
        return train_loader

    def compute_batch_loss(self, batch):
        cl_sidx, aug_op, inputs, labels = batch
        batch_size = cl_sidx.size(0)
        inputs, labels = self.prepare_batch((inputs, labels))
        cl_sidx = cl_sidx.to(self.args.device)
        global_embeddings = self.model.compute_global_embeddings()
        sess_embeddings = self.model.compute_sess_embeddings(
            global_embeddings, inputs, num_base_sessions=batch_size, aug_op=aug_op
        )
        # (batch_size, item_feats)
        main_sess_embeddings = sess_embeddings[:batch_size]
        main_labels = labels[:batch_size]
        logits = main_sess_embeddings @ global_embeddings[1:].t()
        loss = th.nn.functional.cross_entropy(logits, main_labels)
        # (batch_size, num_neg_sessions + 1, item_feats)
        cl_sess_embeddings = sess_embeddings[cl_sidx]
        similarity = th.nn.functional.cosine_similarity(
            cl_sess_embeddings, main_sess_embeddings.unsqueeze(1), dim=-1
        )
        temperature = 0.5
        cl_loss = th.nn.functional.cross_entropy(
            similarity / temperature, labels.new_zeros(batch_size)
        )
        return loss + self.args.beta * cl_loss

    @cached_property
    def collate_fn(self):
        return pad_seqs_and_shift_labels


experiment = Experiment(args)
logging.debug(args)
experiment.run()
