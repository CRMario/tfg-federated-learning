import random as rand
import numpy as np
from utils.config import *

def split_data_by_client(train_data, train_labels, test_data, test_labels, config):
    n_clients = config["n_clients"]
    client_names = [f"client_{i}" for i in range(n_clients)]
    split_method = config["split_method"]
    c = config["C"] 
    alpha = config["alpha"]
    subset = config["subset"]
    balanced = (split_method == "uniform")

    train_labels = np.array(train_labels).flatten()
    test_labels = np.array(test_labels).flatten()

    # initialize the client data
    client_names = [f"client_{i}" for i in range(n_clients)]
    client_data = {name: {"train": {}, "test": {}} for name in client_names}

    # get a part of the dataset
    train_data, train_labels = get_proportional_subset(train_data, train_labels, subset, balanced)
    test_data, test_labels = get_proportional_subset(test_data, test_labels, subset, balanced)

    # list of indices per client
    client_train_ids = [[] for _ in range(n_clients)]
    client_test_ids = [[] for _ in range(n_clients)]

    if split_method == "qbli":
        split_qbli(client_train_ids, client_test_ids, train_labels, test_labels, n_clients, c)
    else:
        split_dirichlet_or_stratified(client_train_ids, client_test_ids, train_labels, test_labels, n_clients, alpha, split_method)

    # now we have the ids per client so we can assign them
    for i, name in enumerate(client_names):
        train_assigned_indices = client_train_ids[i]
        np.random.shuffle(train_assigned_indices)
        test_assigned_indices = client_test_ids[i]
        np.random.shuffle(test_assigned_indices)

        client_data[name]["train"] = {
            "x": [train_data[i] for i in train_assigned_indices],
            "y": [train_labels[i] for i in train_assigned_indices]
        }

        print(len(client_data[name]["train"]["y"]))

        client_data[name]["test"] = {
            "x": [test_data[i] for i in test_assigned_indices],
            "y": [test_labels[i] for i in test_assigned_indices]
        }

        print(len(client_data[name]["test"]["y"]))
            
    return client_data

# TODO: revisar python3 -m src.generate-data --n_clients 8 --dataset local --subset 1 --split_method dirichlet --alpha 0.1
def split_dirichlet_or_stratified(client_train_ids,client_test_ids,train_labels, test_labels, n_clients, alpha, split_method):
    for label in range(len(np.unique(train_labels))):
        train_ids_of_label = np.where(train_labels == label)[0]
        test_ids_of_label = np.where(test_labels == label)[0]
        np.random.shuffle(train_ids_of_label)
        np.random.shuffle(test_ids_of_label)
        proportions = (np.random.dirichlet([alpha] * n_clients) if split_method == "dirichlet" else np.full(n_clients, 1.0 / n_clients))
        # convert to image count. each position is the count assigned to each client
        train_counts = (proportions * len(train_ids_of_label)).astype(int)
        test_counts = (proportions * len(test_ids_of_label)).astype(int)

        train_pos = 0
        test_pos = 0
        for i in range(n_clients):
            n_train_samples = train_counts[i]
            n_test_samples = test_counts[i]
            train_delim = train_pos + n_train_samples
            test_delim = test_pos + n_test_samples
            client_train_ids[i].extend(train_ids_of_label[train_pos:train_delim])
            client_test_ids[i].extend(test_ids_of_label[test_pos:test_delim])
            train_pos += n_train_samples
            test_pos += n_test_samples

"""Pathological quantity based label imbalance controled by C.
Implementation from https://doi.org/10.1109/ICDE53745.2022.00077"""
def split_qbli(client_train_ids,client_test_ids,train_labels,test_labels,n_clients,c):

    n_labels = len(np.unique(train_labels))

    # "We first randomly assign k different label IDs to each party" (in this case c, #C = k)
    # start by assigning different id labels to each client but once the pool runs out of labels
    # we have to refill it. For example if we have 10 clients, 10 labels and we set c = 2, we can
    # only have 5 clients with totally different labels before we have to refill the pool and repeat labels
    # assign C classes to each clieant
    pool = [i for i in range(n_labels)]
    np.random.shuffle(pool)
    client_classes = {i: [] for i in range(n_clients)}
    for i in range(n_clients):
        # if the pool runs out of enough labels to draw refill it
        if len(pool) < c:
            refill_pool = [i for i in range(n_labels)]
            np.random.shuffle(refill_pool)
            pool.extend(refill_pool) # use extend because we still want to assign the remaining labels
        # take the first c labels of the pool 
        chosen = [pool.pop(0) for _ in range(c)]
        client_classes[i] = chosen

    # "Then, for the samples of each label, we randomly and equally divide them into the parties which own the label."
    # find the clients that own the label
    label_to_clients = {l: [] for l in range(n_labels)}
    for client, labels in client_classes.items():
        for l in labels:
            label_to_clients[l].append(client)
    # divide the labels between the parties that own them
    for label in range(n_labels):
        # shuffle the label indices
        train_ids_of_label = np.where(train_labels == label)[0]
        test_ids_of_label = np.where(test_labels == label)[0]

        np.random.shuffle(train_ids_of_label)
        np.random.shuffle(test_ids_of_label)

        # get the owners of the label
        label_owners = label_to_clients[label]
        n_owners = len(label_owners)

        if n_owners > 0:
            train_splits = np.array_split(train_ids_of_label, n_owners)
            test_splits = np.array_split(test_ids_of_label, n_owners)
            for i, owner in enumerate(label_owners):
                client_train_ids[owner].extend(train_splits[i])
                client_test_ids[owner].extend(test_splits[i])

def get_proportional_subset(data, labels, fraction, balance=False):
    unique_labels = np.unique(labels)
    subset_indices = []

    # if the split method wants to keep an equal amount of samples of each class
    # then we can do it while getting the subset of the original dataset
    if balance:
        total_samples = int(len(data) * fraction)
        samples_per_class = total_samples // len(unique_labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n = min(len(label_indices), samples_per_class)
            chosen = np.random.choice(label_indices, n, replace=False)
            subset_indices.extend(chosen)
    else: # keep the original distribution before stratified or non-iid splits
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n = int(len(label_indices) * fraction)
            chosen = np.random.choice(label_indices, n, replace=False)
            subset_indices.extend(chosen)
                
    return [data[i] for i in subset_indices], labels[subset_indices]

"""Splits data into train and test following the ratio given as a parameter
to the configuration in generate-data.py. Sometimes you will not need to use
this method since a lot of datasets already have a train and test split."""
def split_train_test(data, labels, train_ratio):
    unique_labels = np.unique(labels)
    train_data, train_labels, test_data, test_labels = [], [], [], []
    # going to assume the labels could be stored not in order so this method
    # could also be used for other cases other than local datasets
    for label in unique_labels:
        # find the labels
        label_positions = np.where(np.array(labels) == label)[0]

        # split delimitator
        train_del = int(train_ratio * len(label_positions))

        train_positions = label_positions[:train_del]
        test_positions = label_positions[train_del:]

        train_data.extend([data[i] for i in train_positions])
        train_labels.extend([labels[i] for i in train_positions])
        test_data.extend([data[i] for i in test_positions])
        test_labels.extend([labels[i] for i in test_positions])

    return train_data, train_labels, test_data, test_labels