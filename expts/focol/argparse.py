from expts.argparse import ArgumentParserWithCommonArgs


def make_parser():
    parser = ArgumentParserWithCommonArgs()
    parser.add_argument(
        '--num-gnn-layers',
        type=int,
        default=3,
        help='the number of layers in FOGCN'
    )
    parser.add_argument(
        '--num-enc-layers',
        type=int,
        default=3,
        help='the number of transformer layers in the session encoder'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.03,
        help='the scaling factor of constrastive learning loss'
    )
    parser.add_argument(
        '--num-neg-sessions',
        type=int,
        default=32,
        help='the number of negative sessions',
    )
    return parser
