import argparse
import json
import os
import pickle
import numpy as np
from utils.config import *
from src.data_preparation.load_dataset import load_local, load_bloodmnist, load_mnist
from src.config.constants import *

LOAD_DATASET = {
    "local": load_local,
    "bloodmnist": load_bloodmnist,
    "mnist": load_mnist,
}

def main():
    parser = argparse.ArgumentParser(description="A parser for dataset splitting.")

    # Client count
    parser.add_argument("--n_clients",type=int,required=True,help="Number of clients in the federated learning settings")

    # Choose the dataset
    parser.add_argument("--dataset", type=str, choices=["local", "bloodmnist","mnist","cifar"], default="local",
                        help="Choose 'local' for your local folder-based images, or 'bloodmnist' or 'mnist' for an automated download. In " \
                        "case of bloodmnist & mnist you can specify the subset of the data to be used.")

    parser.add_argument("--subset",type=float, default=1.0,
                        help="Controls the proportion of the subset from the downloaded subset.")

    # Data arguments
    parser.add_argument("--train",type=float, default=0.8,
                        help="Controls the proportion of train data.")
    parser.add_argument("--split_method", type=str, choices=["uniform","orig-dist","dirichlet","qbli"],
                        default="orig-dist",help="Choose how the data will be split across clients.")
    parser.add_argument("--alpha",type=float, default=None,
                        help="Controls the heterogeneity for splitting. Must be specified and is only useful if split_method is dirichlet")
    parser.add_argument("--C",type=int, default=None,
                        help="Number of labels to assign to each client. Must be specified and is only useful if split_method is qbli (quantity based label imbalance)")

    # Paths
    parser.add_argument("--data_path",type=str,default="./data/raw")
    parser.add_argument("--save_path", type=str, default="./data/processed")

    # Randomize seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Save the arguments
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    config = load_config("./data/processed/config.json")

    splits, label_mappings = LOAD_DATASET.get(args.dataset,load_local)(config, args.seed)

    with open(os.path.join(args.save_path, 'label_mappings.json'), 'w') as f:
        json.dump(label_mappings, f, indent=4)

    save_path = config["save_path"]
    # Create a pickle file to store the splits
    pickle_file = os.path.join(save_path, "splits.pkl")

    #Dump the hospital data in a pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()