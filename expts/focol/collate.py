from collections import defaultdict
import numpy as np
import torch as th
from .aug_op import AugOp


def pad_seqs_and_shift_labels(samples):
    seqs, labels = zip(*samples)
    labels = th.LongTensor(labels) - 1
    tensor_seqs = list(map(lambda seq: th.LongTensor(np.flip(seq).copy()), seqs))
    padded_seqs = th.nn.utils.rnn.pad_sequence(tensor_seqs, batch_first=True)
    lens = th.LongTensor(list(map(len, seqs)))
    inputs = padded_seqs, lens
    return inputs, labels


class TrainLoaderCollateFn:
    def __init__(self, train_data, num_items, num_neg_sessions, **kwargs):
        self.num_items = num_items
        self.num_neg_sessions = num_neg_sessions
        self.iid2samples = defaultdict(list)
        for seq, label in zip(*train_data):
            self.iid2samples[label].append((seq, label))
        self.iid_list = list(self.iid2samples.keys())
        self.aug_ops = [AugOp.DROPOUT, AugOp.CORRUPT, AugOp.INSERT, AugOp.IDENTITY]

    def get_one_aug_sample(self, label, aug_op):
        sample_idx = np.random.choice(len(self.iid2samples[label]))
        sample = self.iid2samples[label][sample_idx]
        seq, label = sample
        if aug_op == AugOp.INSERT:
            insert_idx = np.random.choice(len(seq) + 1)
            aug_seq = np.insert(seq, insert_idx, 0)
        else:
            aug_seq = seq
        return aug_seq, label

    def get_neg_samples(self, cl_samples, pos_label, num_neg_samples, aug_op):
        neg_sidx = [
            idx for idx, (_, label) in enumerate(cl_samples) if label != pos_label
        ]
        # not enough neg samples in existing cl_samples
        if len(neg_sidx) < num_neg_samples:
            more_neg_samples = []
            for _ in range(num_neg_samples - len(neg_sidx)):
                while True:
                    iid = np.random.choice(self.iid_list)
                    if iid != pos_label:
                        sample = self.get_one_aug_sample(iid, aug_op)
                        more_neg_samples.append(sample)
                        break
            neg_sidx.extend(
                range(len(cl_samples),
                      len(cl_samples) + len(more_neg_samples))
            )
            cl_samples.extend(more_neg_samples)
        sampled_neg_sidx = np.random.choice(neg_sidx, num_neg_samples, replace=False)
        return sampled_neg_sidx

    def __call__(self, samples):
        batch_size = len(samples)
        aug_op = np.random.choice(self.aug_ops)
        cl_samples = []
        # session indices in all_samples.
        cl_sidx = th.empty((batch_size, self.num_neg_sessions + 1), dtype=th.long)
        # get one positive augmented sample for each sample.
        for _, label in samples:
            pos_sample = self.get_one_aug_sample(label, aug_op)
            cl_samples.append(pos_sample)
        cl_sidx[:, 0] = th.arange(batch_size, dtype=th.long)
        for i, (_, label) in enumerate(samples):
            neg_sidx = self.get_neg_samples(
                cl_samples, label, self.num_neg_sessions, aug_op
            )
            cl_sidx[i, 1:] = th.tensor(neg_sidx, dtype=th.long)
        all_samples = list(samples)
        all_samples.extend(cl_samples)
        sidx = cl_sidx + batch_size
        inputs, labels = pad_seqs_and_shift_labels(all_samples)
        return sidx, aug_op, inputs, labels
