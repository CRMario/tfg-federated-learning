import numpy as np
from pathlib import Path
from src.data_preparation.split_data import split_data_by_client, split_train_test
from medmnist import BloodMNIST
from torchvision import datasets
from src.config.constants import *
from utils.config import load_config

def load_images(data_path):
    """
    Loads images and labels from the dataset in data_path.

    The directory is expected to have the following format:
        data_path/
            class_1/
                img1.jpg
                img2.png
                ...
            class_2/
                img3.jpeg
                img4.jpg
                ...
    
    Parameters
    ----------
    data_path : str
        String with the path to the directory that contains the image directories.

    Returns
    -------
    result: dict[str,List[str]]
        A dictionary mapping each class label to a list with the paths of the
        images that belong to the class.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {path}")
    
    # get all label names (directory names in the data_path directory)
    label_names = sorted([d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')])

    # create the mappings
    id_to_label = {i: name for i, name in enumerate(label_names)}
    label_to_id = {name: i for i, name in enumerate(label_names)}

    data = []
    labels = []

    for label in label_names:
        label_id = label_to_id[label]
        label_imgs_dir = path / label
        for image_path in label_imgs_dir.iterdir():
            if image_path.suffix.lower() in EXTENSIONS:
                data.append(str(image_path))
                labels.append(label_id)

    return data, labels, {"label_to_id": label_to_id,"id_to_label": id_to_label}

"""Loads the local dataset, divides it in train and test and
splits it across clients"""
def load_local(config):
    # return the labels already as integers and handle the real name label mapping
    # when finding the images in the corresponding labeled directories
    data, labels, label_mappings = load_images(config["data_path"])

    train_data, train_labels, test_data, test_labels = split_train_test(data, labels, config["train"])

    splits = split_data_by_client(train_data=train_data,
                                  train_labels=train_labels, 
                                  test_data=test_data, 
                                  test_labels=test_labels, 
                                  config=config)

    return splits, label_mappings

"""Loads the BloodMNIST dataset and splits across clients"""
def load_bloodmnist(config):
    train_set = BloodMNIST(split='train',root='./data/bloodmnist', download=True)
    test_set = BloodMNIST(split='test', root='./data/bloodmnist', download=True)

    label_mappings = {
        "label_to_id": {label: i for i, label in train_set.info['label'].items()},
        "id_to_label": {i: label for i, label in train_set.info['label'].items()}
    }

    splits = split_data_by_client(train_data=train_set.imgs,
                                  train_labels=train_set.labels,
                                  test_data=test_set.imgs,
                                  test_labels=test_set.labels,
                                  config=config)
        
    return splits, label_mappings

"""Loads the MNIST dataset and splits across clients"""
def load_mnist(config):
    train_set = datasets.MNIST(root='./data', train=True, download=True)
    test_set = datasets.MNIST(root='./data', train=False, download=True)

    label_mappings = {
        "label_to_id": {label: i for i, label in enumerate(train_set.classes)},
        "id_to_label": {i: label for i, label in enumerate(train_set.classes)}
    }

    splits = split_data_by_client(train_data=np.array(train_set.data),
                                  train_labels=train_set.targets,
                                  test_data=np.array(test_set.data),
                                  test_labels=test_set.targets,
                                  config=config)
        
    return splits, label_mappings