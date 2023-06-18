import sys

if sys.argv[1].startswith('-'):
    print(
        f'usage: {sys.argv[0]} [-h] SCRIPT [ARGUMENTS...]\n\n'
        '  -h, --help    show this help message and exit\n'
        '  SCRIPT        a Python script\n'
        '  ARGUMENTS     arguments for the python script\n'
    )
    sys.exit(0)

import importlib
from pathlib import Path

while sys.argv[1][-3:] == '.py':
    module = '.'.join(Path(sys.argv[1][:-3]).parts)
    print(module)
    sys.argv = sys.argv[1:]
    importlib.import_module(module)
