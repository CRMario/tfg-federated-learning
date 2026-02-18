import argparse
import json
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from utils.config import *
from torchvision import datasets
from src.data_preparation.load_dataset import load_images
from src.data_preparation.split_data import split_data_by_client, split_cifar_by_client
from src.config.constants import *

# Run as python -m src.generate-data
def main():
    parser = argparse.ArgumentParser(description="A parser for dataset splitting.")

    # Client count
    parser.add_argument("--n_clients",type=int,default=3)

    # Choose the dataset
    parser.add_argument("--dataset", type=str, choices=["local", "cifar10"], default="local",
                        help="Choose 'local' for your local folder-based images or 'cifar10' for an automated download. In " \
                        "case of cifar10 only a 10 percent of the original dataset will be used")

    parser.add_argument("--subset",type=float, default=0.1,
                        help="Controls the proportion of the subset from the downloaded subset.")

    # Data arguments
    parser.add_argument("--train",type=float, default=0.8,
                        help="Controls the proportion of train data.")
    parser.add_argument("--split_method", type=str, choices=["iid","non-iid"],
                        default="iid",help="Choose how the data will be split across clients.")
    parser.add_argument("--alpha",type=float, default=0.5,
                        help="Controls the heterogeneity for splitting. Only useful if split_method is non-iid")

    # Paths
    parser.add_argument("--data_path",type=str,default="./data/raw")
    parser.add_argument("--save_path",type=str,default="./data/processed")

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Save the arguments
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
        test_set = datasets.CIFAR10(root='./data/cifar10', train=False, download=True)
        splits = split_cifar_by_client(train_set, test_set)

        label_mappings = {
            "label_to_id": {label: i for i, label in enumerate(train_set.classes)},
            "id_to_label": {i: label for i, label in enumerate(train_set.classes)}
        }
    else:
        images = load_images(args.data_path)
        unique_labels = sorted(list(images.keys()))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for label, i in label_to_id.items()}

        label_mappings = {
            "label_to_id": label_to_id,
            "id_to_label": id_to_label
        }

        # Split the images amongst clients
        splits = split_data_by_client(images)

    with open(os.path.join(args.save_path, 'label_mappings.json'), 'w') as f:
        json.dump(label_mappings, f, indent=4)

    config = load_config("./data/processed/config.json")
    save_path = config["save_path"]

    # Create a pickle file to store the splits
    pickle_file = os.path.join(save_path, "splits.pkl")

    #Dump the hospital data in a pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump(splits, f)

if __name__ == "__main__":
    main()