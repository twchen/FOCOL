from collections import defaultdict
import torch as th


def evaluate_next_item_rec(model, data_loader, prepare_batch, Ks=[20]):
    model.eval()
    vals = defaultdict(float)
    max_K = max(Ks)
    num_samples = 0
    with th.no_grad():
        for batch in data_loader:
            inputs, labels = prepare_batch(batch)
            logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = th.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = th.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                vals[f'HR@{K}'] += hit_ranks.numel()
                vals[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                vals[f'NDCG@{K}'] += th.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in vals:
        vals[metric] /= num_samples
    return vals
