import time
import pickle
import logging
from collections import defaultdict
import torch as th
from torch.utils.data import DataLoader
from utils.evaluate import evaluate_next_item_rec
from utils.cached_property import cached_property


class Dataset(th.utils.data.Dataset):
    def __init__(self, data):
        self.seqs, self.labels = data

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


class ExperimentBase:
    def __init__(self, args):
        self.args = args

    def read_dataset(self):
        def load_pickle(filepath):
            with open(filepath, 'rb') as file:
                return pickle.load(file)

        self.args.train_sessions = load_pickle(
            self.args.dataset_dir / 'all_train_seq.txt'
        )
        train_data = load_pickle(self.args.dataset_dir / 'train.txt')
        valid_filepath = self.args.dataset_dir / 'valid.txt'
        if valid_filepath.exists():
            self.args.valid_data = load_pickle(valid_filepath)
        test_data = load_pickle(self.args.dataset_dir / 'test.txt')

        train_seqs, train_labels = train_data
        seq_max_iid = max(map(max, train_seqs))
        label_max_iid = max(train_labels)
        self.args.num_items = max(seq_max_iid, label_max_iid)  # the IDs start from 1
        self.args.max_len = max([
            max(map(len, seqs)) for seqs, _ in [train_data, test_data]
        ]) + 1
        self.args.train_data = train_data
        self.args.test_data = test_data

    @cached_property
    def train_loader(self):
        return DataLoader(
            Dataset(self.args.train_data),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

    @cached_property
    def valid_loader(self):
        if 'valid_data' in self.args:
            return DataLoader(
                Dataset(self.args.valid_data),
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers,
            )
        else:
            return None

    @cached_property
    def test_loader(self):
        return DataLoader(
            Dataset(self.args.test_data),
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
        )

    @cached_property
    def model(self):
        return self.args.ModelClass(**self.args).to(self.args.device)

    @cached_property
    def collate_fn(self):
        raise NotImplementedError

    @cached_property
    def optimizer(self):
        optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        return optimizer

    def prepare_batch(self, batch):
        inputs, labels = batch
        inputs_gpu = [x.to(self.args.device) for x in inputs]
        labels_gpu = labels.to(self.args.device)
        return inputs_gpu, labels_gpu

    def compute_batch_loss(self, batch):
        inputs, labels = self.prepare_batch(batch)
        logits = self.model(*inputs)
        loss = th.nn.functional.cross_entropy(logits, labels)
        return loss

    def train(self):
        self.epoch = 0
        self.batch = 0
        self.report_scores = defaultdict(float)
        self.max_scores = defaultdict(lambda: float('-inf'))
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        for epoch in range(self.args.epochs):
            logging.warning(f'Epoch {self.epoch}:')
            self.model.train()
            train_ts = time.time()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.compute_batch_loss(batch)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / self.args.log_interval
                if self.batch > 0 and self.batch % self.args.log_interval == 0:
                    logging.info(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                self.batch += 1

            if self.epoch + 1 < self.args.epochs and self.epoch % self.args.eval_interval != 0:
                self.epoch += 1
                continue
            # log at the last epoch or when self.epoch divides eval_interval

            train_te = time.time()
            logging.debug(f'Total training time: {train_te - train_ts:.2f}s')
            eval_ts = time.time()
            self.comp_scores, self.test_scores = self.evaluate()
            eval_te = time.time()
            logging.debug(f'Total evaluation time: {eval_te - eval_ts:.2f}s')

            any_better_score = False
            for metric in self.comp_scores:
                if self.comp_scores[
                        metric] > self.max_scores[metric] + self.args.patience_delta:
                    self.max_scores[metric] = self.comp_scores[metric]
                    self.report_scores[metric] = self.test_scores[metric]
                    any_better_score = True
            if not any_better_score:
                bad_counter += 1
                if bad_counter == self.args.patience:
                    break
            else:
                bad_counter = 0
            self.epoch += 1
            epoch_te = time.time()
            t += epoch_te - train_te
        logging.info(self.scores_to_str({'Report': self.report_scores}))
        return self.report_scores

    def evaluate(self):
        self.test_scores = self.evaluate_on_data_loader(self.test_loader)
        scores_dict = {}
        if self.valid_loader is not None:
            comp_scores = self.evaluate_on_data_loader(self.valid_loader)
            scores_dict['Valid'] = comp_scores
        else:
            comp_scores = self.test_scores
        scores_dict['Test'] = self.test_scores
        logging.warning(self.scores_to_str(scores_dict))
        return comp_scores, self.test_scores

    def evaluate_on_data_loader(self, data_loader):
        return evaluate_next_item_rec(
            self.model, data_loader, self.prepare_batch, self.args.Ks
        )

    def scores_to_str(self, scores_dict):
        metrics = next(iter(scores_dict.values()))  # metric names
        widths = [max(5, len(metric)) for metric in metrics]
        header = 'Metric\t' + '\t'.join([
            f'{metric:>{width}}' for metric, width in zip(metrics, widths)
        ])
        lines = [header]
        for name, scores in scores_dict.items():
            values = '\t'.join([
                f'{round(scores[metric] * 100, 2):>{width}.2f}'
                for metric, width in zip(metrics, widths)
            ])
            line = name + '\t' + values
            lines.append(line)
        string = '\n'.join(lines)
        return string

    def run(self):
        self.read_dataset()
        logging.debug(self.model)
        self.train()
