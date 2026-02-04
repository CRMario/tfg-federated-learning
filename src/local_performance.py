from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from src.data_preparation.load_dataset import load_images
from src.data_preparation.split_data import split_data_by_client
from src.config.constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from random import shuffle

class ImageDataset(Dataset):

    transforms = Compose([Resize((224,224)), ToTensor(), 
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        unique_labels = sorted(list(set(labels)))
        # for converting the labels into integers to use them in the dataloader
        self.id_label = {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image_path = self.images[id]
        # only load the image when necessary (when we get the image)
        image = Image.open(image_path).convert("RGB")
        image = self._apply_transforms(image)
        label = self.labels[id]
        id_for_label = self.id_label[label]

        label = torch.tensor(id_for_label, dtype=torch.long)
        # skorch expects a tuple for their fit method
        return (image,id_for_label)
    
    def _apply_transforms(self,image):
        return self.transforms(image)

class CNN(nn.Module):

    def __init__(self, n_conv_layers=3, n_fc_layers=3, starting_filters=16, n_labels=3, kernel_size=3):
        super(CNN, self).__init__()
        conv_layers = []
        fc_layers = []
        in_channels = 3
        out_channels = starting_filters

        for _ in range(n_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2,2))
            in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*conv_layers)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        in_fc = in_channels
        out_fc = 128
        for _ in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(in_fc,out_fc))
            in_fc = out_fc
            out_fc //= 2
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.5))

        fc_layers.append(nn.Linear(in_fc,n_labels))

        self.classification = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.classification(x)
        return x

""" Test model performance on the dataset as if the data scientist was
allowed to see the information. This will let us compare accuracy against
privacy.
"""
def main():

    # Load the images: 70% train 30% test
    train_imgs, train_labels, test_imgs, test_labels = [], [], [], []
    for label, image_list in load_images(DATA_PATH).items():
        shuffle(image_list)
        train_del = int(0.7 * len(image_list))
        train = image_list[:train_del]
        test = image_list[train_del:]
        train_imgs.extend(train)
        test_imgs.extend(test)
        train_labels.extend([label] * len(train))
        test_labels.extend([label] * len(test))

    train_ds = ImageDataset(train_imgs, train_labels)
    test_ds = ImageDataset(test_imgs, test_labels)

    model = NeuralNetClassifier(
        module=CNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        lr=0.001,
        batch_size=32,
        max_epochs=10,
        iterator_train__num_workers=4
    )

    params = {
        # CNN parameters
        'module__n_conv_layers': [2, 3, 4],
        'module__n_fc_layers': [2, 3],
        'module__starting_filters': [16]#, 32],
        #'module__kernel_size': [3, 5],
        # Optimization parameters
        #'lr': [0.001, 0.01]
    }

    y_train_indices = np.array([train_ds.id_label[label] for label in train_labels])

    grid_search = GridSearchCV(model, params, refit=True, cv=3)
    grid_search.fit(train_ds, y=y_train_indices)

    print(f"Best Score: {grid_search.best_score_}")
    print(f"Best Params: {grid_search.best_params_}")


if __name__ == "__main__":
    main()