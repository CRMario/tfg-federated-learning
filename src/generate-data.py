import argparse
import json
import os
import pickle
from utils.config import *
from src.data_preparation.load_dataset import load_images
from src.data_preparation.split_data import split_data_by_client
from src.config.constants import *

# Run as python -m src.generate-data
def main():
    parser = argparse.ArgumentParser(description="A parser for dataset splitting.")

    # Client count
    parser.add_argument("--n_clients",type=int,default=3)

    # Data arguments
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

    # Load the images of the dataset
    images = load_images(DATA_PATH)

    # Split the images amongst the hospitals
    hospitals_splits = split_data_by_client(images)

    config = load_config("./data/processed/config.json")
    save_path = config["save_path"]

    # Create a pickle file to store the splits
    pickle_file = os.path.join(save_path, "splits.pkl")

    # 3. Dump the dictionary
    with open(pickle_file, "wb") as f:
        pickle.dump(hospitals_splits, f)

if __name__ == "__main__":
    main()