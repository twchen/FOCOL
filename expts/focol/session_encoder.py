import torch as th
from torch import nn
from .aug_op import AugOp


class SessionEncoder(nn.Module):
    def __init__(self, item_feats, num_layers, max_len):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                item_feats,
                nhead=1,
                dim_feedforward=2 * item_feats,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.rev_pos_embedding = nn.Embedding(max_len, item_feats)

    def forward(
        self, embedding, rev_padded_seqs, lens, num_base_sessions=None, aug_op=None
    ):
        emb_seqs = embedding[rev_padded_seqs]
        batch_size, max_len, item_feats = emb_seqs.size()
        if self.training:
            assert aug_op is not None
            if aug_op == AugOp.CORRUPT:
                zeros = emb_seqs.new_zeros((num_base_sessions, max_len, item_feats))
                noise = th.normal(
                    mean=0,
                    std=0.1,
                    size=(batch_size - num_base_sessions, max_len, item_feats),
                    dtype=emb_seqs.dtype,
                    device=emb_seqs.device
                )
                noise = th.cat([zeros, noise], dim=0)
                emb_seqs = emb_seqs + noise
            elif aug_op == AugOp.DROPOUT:
                emb_seqs = th.nn.functional.dropout(emb_seqs, p=0.1, training=True)
        rev_pos = th.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len)
        rev_pos_emb = self.rev_pos_embedding(rev_pos)
        emb_seqs = emb_seqs + rev_pos_emb
        padding_mask = rev_pos >= lens.unsqueeze(-1)
        emb_seqs = self.transformer(emb_seqs, src_key_padding_mask=padding_mask)
        sess_emb = emb_seqs[th.arange(batch_size), lens.new_zeros(batch_size)]
        return sess_emb
