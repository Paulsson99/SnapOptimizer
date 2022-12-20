from argparse import ArgumentParser
import json
from datetime import datetime
import numpy as np
from pathlib import Path

from SnapOptimizer.paths import defaults
from SnapOptimizer.encodings import read_encoding


def parse_args():
    parser = ArgumentParser("SNAP optimization")
    parser.add_argument('-e', '--encoding', 
        help="JSON file with the encoding to use. Each key is a qubit state. The values should be a list of fock states to encode that qubit state",
        default=defaults('default_encoding.json'),
        type=Path
    )
    parser.add_argument('-g', '--gates', action='extend', nargs='+', help="The gatets to run the optimization for", required=True)
    parser.add_argument('--epochs', help='Number of epochs to run the optimization for', type=int, default=3000)
    parser.add_argument('-N', help='Number of fockstates in the Hilbert space', type=int, action='extend', nargs='+', required=True)
    parser.add_argument('-n', '--n-gates', help='Number of SNAP gates to use', type=int, action='extend', nargs='+', required=True)
    parser.add_argument('-s', '--show-figures', action='store_true', help="When this flag is present the figures pop up on screen when done")
    parser.add_argument('-o', '--output-folder', help='Folder to save all data to', default=None, type=Path)
    parser.add_argument('-a', '--averages', help='Run the same optimization multiple times', default=1, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup the output folder
    if args.output_folder is None:
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        output_folder = Path('data') / f"data_{timestamp}"
    else:
        output_folder = args.output_folder

    encoding = read_encoding(args.encoding)

    # Only do heavy imports when it is needed
    from SnapOptimizer.optimization.automation import optimize_SNAP_gates

    optimize_SNAP_gates(
        encoding=encoding,
        gates=args.gates,
        n_gates=args.n_gates,
        Ns=args.N,
        epochs=args.epochs,
        output_folder=output_folder,
        show_figure=args.show_figures,
        averages=args.averages
    )
