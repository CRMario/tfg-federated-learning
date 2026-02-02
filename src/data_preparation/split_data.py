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
