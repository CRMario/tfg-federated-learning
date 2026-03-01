import random as rand
import numpy as np
from utils.config import *

def split_data_by_client(images,seed=42):
    """
    Splits the given images between the clients by shuffling the lists
    of images. Each client is given an equal amount of images of each label.

    Parameters
    ----------
    images : Dict[str,List[Image]]
        A dictionary where the key is the label and the value is a
        list containing all the images that were assigned to that label.

    seed : int
        An integer that initializes the rand.
        
    Returns
    -------
    data_splits: Dict[str,Dict[str,List[str]]]
        A dictionary mapping each client to a dictionary that
        maps each label to a list of images that have that label.
    """

    config = load_config("./data/processed/config.json")
    n_clients = config["n_clients"]
    hospital_names = [f"hospital_{i}" for i in range(n_clients)]
    split_method = config["split_method"]
    alpha = config["alpha"]
    train_ratio = config["train"]

    np.random.seed(seed)
    rand.seed(seed)

    hospital_data = {name: {"train": {}, "test": {}} for name in hospital_names}
    
    for label, imgs in images.items():
        # Shuffle the array of images
        rand.shuffle(imgs)

        proportions = (np.random.dirichlet([alpha] * n_clients) if split_method == "non-iid" else np.full(n_clients, 1.0 / n_clients))
        counts = (proportions * len(imgs)).astype(int) # Convert to image count, example [680,2380,340] 
        # ensure at least 1 image per client
        if len(imgs) >= n_clients:
            for i in range(len(counts)):
                if counts[i] == 0:
                    counts[i] = 1
                    # Subtract that 1 from the client who has the most
                    counts[np.argmax(counts)] -= 1
            
        start = 0
        for hospital, count in enumerate(counts):
            h_name = hospital_names[hospital]
            end = start + count

            client_images = imgs[start:end]
            
            # ensure at least 1 image goes to train if count > 0
            if len(client_images) > 0:
                train_count = max(1, int(len(client_images) * train_ratio))
            else:
                train_count = 0

            hospital_data[h_name]["train"][label] = client_images[:train_count]
            hospital_data[h_name]["test"][label] = client_images[train_count:]
            start = end
    
    return hospital_data

def split_bloodmnist_by_client(train_set, test_set):
    config = load_config("./data/processed/config.json")
    n_clients = config["n_clients"]
    client_names = [f"client_{i}" for i in range(n_clients)]
    split_method = config["split_method"]
    c = config["C"] 
    alpha = config["alpha"]
    subset = config["subset"]
    balanced = (split_method == "iid")

    # initialize the client data
    client_names = [f"client_{i}" for i in range(n_clients)]
    client_data = {name: {"train": {}, "test": {}} for name in client_names}

    def get_proportional_subset(dataset, fraction, balance=False):
        data = dataset.imgs
        labels = np.array(dataset.labels).flatten()
        unique_labels = np.unique(labels)
        subset_indices = []

        if balance: #iid
            total_samples = int(len(data) * fraction)
            samples_per_class = total_samples // len(unique_labels)
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                n = min(len(label_indices), samples_per_class)
                chosen = np.random.choice(label_indices, n, replace=False)
                subset_indices.extend(chosen)
        else: #original distribution
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                n = int(len(label_indices) * fraction)
                chosen = np.random.choice(label_indices, n, replace=False)
                subset_indices.extend(chosen)
                
        return data[subset_indices], labels[subset_indices]

    train_data, train_labels = get_proportional_subset(train_set, subset, balanced)
    test_data, test_labels = get_proportional_subset(test_set, subset, balanced)

    # list of indices per client
    client_train_ids = [[] for _ in range(n_clients)]
    client_test_ids = [[] for _ in range(n_clients)]

    if split_method == "qbli": # pathological quantity based label imbalance controled by C
        # implementation from https://doi.org/10.1109/ICDE53745.2022.00077

        # "We first randomly assign k different label IDs to each party" (in this case c, #C = k)
        # start by assigning different id labels to each client but once the pool runs out of labels
        # we have to refill it. For example if we have 10 clients, 10 labels and we set c = 2, we can
        # only have 5 clients with totally different labels before we have to refill the pool and repeat labels
        # assign C classes to each clieant
        pool = [i for i in range(8)]
        np.random.shuffle(pool)
        client_classes = {i: [] for i in range(n_clients)}
        for i in range(n_clients):
            # if the pool runs out of enough labels to draw refill it
            if len(pool) < c:
                refill_pool = [i for i in range(8)]
                np.random.shuffle(refill_pool)
                pool.extend(refill_pool) # use extend because we still want to assign the remaining labels
            # take the first c labels of the pool 
            chosen = [pool.pop(0) for _ in range(c)]
            client_classes[i] = chosen

        # "Then, for the samples of each label, we randomly and equally divide them into the parties which own the label."
        # find the clients that own the label
        label_to_clients = {l: [] for l in range(8)}
        for client, labels in client_classes.items():
            for l in labels:
                label_to_clients[l].append(client)
        # divide the labels between the parties that own them
        for label in range(8):
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

    else: # non-iid and iid
        for label in range(8):
            train_ids_of_label = np.where(train_labels == label)[0]
            test_ids_of_label = np.where(test_labels == label)[0]
            np.random.shuffle(train_ids_of_label)
            np.random.shuffle(test_ids_of_label)
            proportions = (np.random.dirichlet([alpha] * n_clients) if split_method == "non-iid" else np.full(n_clients, 1.0 / n_clients))
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

    # now we have the ids per client so we can assign them
    for i, name in enumerate(client_names):
        train_assigned_indices = client_train_ids[i]
        np.random.shuffle(train_assigned_indices)
        test_assigned_indices = client_test_ids[i]
        np.random.shuffle(test_assigned_indices)

        client_data[name]["train"] = {
            "x": train_data[train_assigned_indices],
            "y": train_labels[train_assigned_indices]
        }

        client_data[name]["test"] = {
            "x": test_data[test_assigned_indices],
            "y": test_labels[test_assigned_indices]
        }
            
    print(client_data)
    return client_data