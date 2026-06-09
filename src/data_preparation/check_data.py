import os
from collections import Counter
from utils.config import load_config
import matplotlib.pyplot as plt
import pickle
import numpy as np

def print_distribution_table(title, client_ids, unique_labels, label_names, counts_dict):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    header = f"{'Client ID':<12}"
    for label in unique_labels:
        header += f" | {label_names[label]:<10}"
    header += " | Total"
    print(header)
    print("-" * len(header))

    for client_id in client_ids:
        row = f"{str(client_id):<12}"
        total_samples = 0
        for label in unique_labels:
            count = counts_dict[client_id].get(label, 0)
            row += f" | {count:<10}"
            total_samples += count
        row += f" | {total_samples}"
        print(row)
    print(f"{'='*70}\n")

def save_distribution_plot(set_name, client_ids, unique_labels, all_labels, counts_dict, config, out_dir):
    split_method = config["split_method"]
    dataset = config["dataset"]
    
    plot_matrix = np.zeros((len(unique_labels), len(client_ids)))
    for c_idx, client_id in enumerate(client_ids):
        for l_idx, label in enumerate(unique_labels):
            plot_matrix[l_idx, c_idx] = counts_dict[client_id].get(label, 0)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(client_ids))
    
    for i, label in enumerate(unique_labels):
        label_name = all_labels[label]
        plt.bar(client_ids, plot_matrix[i, :], bottom=bottom, label=label_name)
        bottom += plot_matrix[i, :]

    title_suffix = f" ({set_name.capitalize()} Set) - {split_method}"
    if split_method == "dirichlet":
        title_suffix += f", alpha = {config['alpha']}"
    elif split_method == "qbli":
        title_suffix += f", #C = {config['C']}"
    
    plt.title(f"Label Distribution Across Clients {title_suffix}")
    plt.xlabel("Client ID")
    plt.ylabel(f"Number of Samples ({set_name.capitalize()})")
    plt.xticks(rotation=45)
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"{split_method}_{dataset}_{set_name}_distribution.png"
    plt.savefig(os.path.join(out_dir, filename))
    print(f"Saved as: {os.path.join(out_dir, filename)}")
    plt.show()

def main():
    with open("./data/processed/splits.pkl", "rb") as f:
        clients_data = pickle.load(f)

    config = load_config("./data/processed/config.json")
    dataset = config["dataset"]
    label_mappings = load_config("./data/processed/label_mappings.json")
    all_labels = {int(k): v for k, v in label_mappings["id_to_label"].items()}
    client_ids = sorted(clients_data.keys())
    unique_labels = sorted(all_labels.keys())

    train_counts = {}
    test_counts = {}

    for client_id in client_ids:
        train_labels = clients_data[client_id]["train"]["y"]
        train_counts[client_id] = Counter(train_labels)
        
        test_labels = clients_data[client_id]["test"]["y"]
        test_counts[client_id] = Counter(test_labels)

    print_distribution_table(f"REPARTO EN TRAIN ({dataset.upper()})", 
                             client_ids, unique_labels, all_labels, train_counts)
    
    print_distribution_table(f"REPARTO EN TEST ({dataset.upper()})", 
                             client_ids, unique_labels, all_labels, test_counts)

    out_dir = "./data/distributions"
    os.makedirs(out_dir, exist_ok=True)

    save_distribution_plot("train", client_ids, unique_labels, all_labels, train_counts, config, out_dir)
    save_distribution_plot("test", client_ids, unique_labels, all_labels, test_counts, config, out_dir)

if __name__ == "__main__":
    main()