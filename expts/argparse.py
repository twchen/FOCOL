import sys
from pathlib import Path
import argparse


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )
        self.optional = self._action_groups.pop()
        self.required = self.add_argument_group('required arguments')
        self._action_groups.append(self.optional)

    def add_argument(self, *args, **kwargs):
        if kwargs.get('required', False):
            self.required.add_argument(*args, **kwargs)
        else:
            super().add_argument(*args, **kwargs)


def str2listof(type):
    def str2list(v):
        return [type(x) for x in v.split(',')]

    return str2list


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ArgumentParserWithCommonArgs(ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(prog=sys.argv[0], **kwargs)
        self.add_argument(
            '--dataset-dir',
            type=Path,
            required=True,
            help='the dataset set directory, e.g. datasets/diginetica',
        )
        self.add_argument(
            '--item-feats',
            type=int,
            default=100,
            help='the number of features in the item embeddings',
        )
        self.add_argument('--batch-size', type=int, default=100, help='the batch size')
        self.add_argument(
            '--epochs', type=int, default=30, help='the number of training epochs'
        )
        self.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
        self.add_argument(
            '--weight-decay',
            type=float,
            default=1e-5,
            help='the weight decay for the optimizer',
        )
        self.add_argument(
            '--log-level',
            choices=['debug', 'info', 'warning', 'error'],
            default='debug',
            help='the log level',
        )
        self.add_argument(
            '--log-interval',
            type=int,
            default=1000,
            help='if log level is at least info, print some information after every this number of iterations',
        )
        self.add_argument(
            '--eval-interval',
            type=int,
            default=1,
            help='evaluate the model on the validation set (if any) and the test set every this number of epochs'
        )
        self.add_argument(
            '--Ks',
            type=str2listof(int),
            default='10,20',
            help='the values of K in evaluation metrics, separated by commas',
        )
        self.add_argument(
            '--patience',
            type=int,
            default=1,
            help='stop training if the performance does not improve in this number of consecutive epochs',
        )
        self.add_argument(
            '--patience-delta',
            type=float,
            default=5e-5,
            help='the minimum difference of performance that is considered as an improvement',
        )
        self.add_argument(
            '--num-workers',
            type=int,
            default=0,
            help='the number of processes for data loaders',
        )
        self.add_argument(
            '-d', '--device', type=int, default=0, help='GPU device index (-1 for CPU)'
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        import logging
        from utils.dotted_dict import DottedDict
        import torch as th

        log_level = getattr(logging, args.log_level.upper(), None)
        logging.basicConfig(format='%(message)s', level=log_level)

        args = DottedDict(vars(args))
        args.device = th.device('cpu' if args.device < 0 else f'cuda:{args.device}')
        return args
