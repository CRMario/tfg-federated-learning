import argparse
import json
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from utils.config import *
from torchvision import datasets
from src.data_preparation.load_dataset import load_images
from src.data_preparation.split_data import split_data_by_client, split_bloodmnist_by_client
from src.config.constants import *
from medmnist import BloodMNIST

def main():
    parser = argparse.ArgumentParser(description="A parser for dataset splitting.")

    # Client count
    parser.add_argument("--n_clients",type=int,default=3)

    # Choose the dataset
    parser.add_argument("--dataset", type=str, choices=["local", "bloodmnist"], default="local",
                        help="Choose 'local' for your local folder-based images or 'bloodmnist' for an automated download. In " \
                        "case of bloodmnist you can specify the subset of the data to be used.")

    parser.add_argument("--subset",type=float, default=1.0,
                        help="Controls the proportion of the subset from the downloaded subset.")

    # Data arguments
    parser.add_argument("--train",type=float, default=0.8,
                        help="Controls the proportion of train data.")
    parser.add_argument("--split_method", type=str, choices=["iid","non-iid","orig-dist","qbli"],
                        default="iid",help="Choose how the data will be split across clients.")
    parser.add_argument("--alpha",type=float, default=None,
                        help="Controls the heterogeneity for splitting. Must be specified and is only useful if split_method is non-iid")
    parser.add_argument("--C",type=int, default=None,
                        help="Number of labels to assign to each client. Must be specified and is only useful if split_method is qbli (quantity based label imbalance)")

    # Paths
    parser.add_argument("--data_path",type=str,default="./data/raw")
    parser.add_argument("--save_path",type=str,default="./data/processed")

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Save the arguments
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.dataset == "bloodmnist":
        train_set = BloodMNIST(split='train',root='./data/bloodmnist', download=True)
        test_set = BloodMNIST(split='test', root='./data/bloodmnist', download=True)
        splits = split_bloodmnist_by_client(train_set, test_set)
        label_mappings = {
            "label_to_id": {label: i for i, label in train_set.info['label'].items()},
            "id_to_label": {i: label for i, label in train_set.info['label'].items()}
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