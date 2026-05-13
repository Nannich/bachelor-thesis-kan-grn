import ctypes
import os

try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import os
import argparse
from src.train import run_training
from src.plotting.visualize import run_visualization
from src.analysis.symbolic import run_extraction
from src.analysis.de import run_de
from src.analysis.grn import run_grn
from src.preprocessing import run_preprocessing

def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--model_dir", type=str, default="./checkpoints/")
    parser.add_argument("--fig_dir", type=str, default="./figures/")

    # Parse dataset
    parser.add_argument("--dataset", type=str, default="paul")

    # Parse command specific arguments
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--model", type=str, choices=["effkan", "pykan", "mlp", "null"], default="effkan")
    parser_train.add_argument("--gene", type=int, default=None) # Choose None to train all

    parser_vis = subparsers.add_parser("visualize")
    parser_vis.add_argument("name", type=str)
    parser_vis.add_argument("gene", type=int)

    parser_sym = subparsers.add_parser("symbolic")
    parser_sym.add_argument("name", type=str)
    parser_sym.add_argument("gene", type=int)
    
    parser_de = subparsers.add_parser("de")
    parser_de.add_argument("name", type=str)
    parser_de.add_argument("--lineage", type=int, default=0)

    parser_grn = subparsers.add_parser("grn")
    parser_grn.add_argument("name", type=str)

    parser_process = subparsers.add_parser("process")


    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    args.model_dir = os.path.expanduser(args.model_dir)
    args.fig_dir = os.path.expanduser(args.fig_dir)

    # Fetch and preprocess the dataset
    adata, pseudotime, weights = run_preprocessing(args)    

    # Run the correct script based on the command
    if args.command == "train":
        run_training(args, adata, pseudotime, weights)
    elif args.command == "visualize":
        run_visualization(args, adata, pseudotime, weights)
    elif args.command == "symbolic":
        run_extraction(args, adata, pseudotime, weights)
    elif args.command == "de":
        run_de(args, adata, pseudotime, weights)
    elif args.command == "grn":
        run_grn(args, adata, pseudotime, weights)
    elif args.command == "process":
        run_preprocessing(args)


if __name__ == "__main__":
    main()