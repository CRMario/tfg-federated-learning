import os
from collections import Counter
from utils.config import load_config
import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():
    
    with open("./data/processed/splits.pkl", "rb") as f:
        clients_data = pickle.load(f)

    config = load_config("./data/processed/config.json")
    split_method = config["split_method"]
    dataset = config["dataset"]
    label_mappings = load_config("./data/processed/label_mappings.json")
    all_labels = {int(k): v for k, v in label_mappings["id_to_label"].items()}
    client_ids = sorted(clients_data.keys())
    label_counts_per_client = {}

    for client_id in client_ids:
        labels = clients_data[client_id]["train"]["y"]
        label_counts_per_client[client_id] = Counter(labels)

    unique_labels = sorted(all_labels.keys())

    # rows are the labels columns are the clients
    plot_matrix = np.zeros((len(unique_labels), len(client_ids)))
    for c_idx, client_id in enumerate(client_ids):
        for l_idx, label in enumerate(unique_labels):
            plot_matrix[l_idx, c_idx] = label_counts_per_client[client_id].get(label, 0)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(client_ids))
    
    for i, label in enumerate(unique_labels):
        label_name = all_labels[label]
        plt.bar(client_ids, plot_matrix[i, :], bottom=bottom, label=label_name)
        bottom += plot_matrix[i, :]

    if (split_method == "orig-dist"):
        plt.title(f"Label Distribution Across Clients {split_method}")
    elif (split_method == "dirichlet"):
        plt.title(f"Label Distribution Across Clients {split_method}, alpha = {config["alpha"]}")
    elif (split_method == "qbli"):
        plt.title(f"Label Distribution Across Clients {split_method}, #C = {config["C"]}")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    out_dir = "./data/distributions"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{split_method}_{dataset}_distribution.png"
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()