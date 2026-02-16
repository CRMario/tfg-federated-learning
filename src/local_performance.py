from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from src.data_preparation.load_dataset import load_images
from src.data_preparation.split_data import split_data_by_client
from src.config.constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from random import shuffle
from collections import Counter

class ImageDataset(Dataset):

    transforms = Compose([Resize((224,224)), ToTensor(), 
                                  Normalize([0.5], [0.5])])
    
    def __init__(self, images, labels, label_hash):
        self.images = images
        self.labels = labels
        self.id_label = label_hash

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image_path = self.images[id]
        # only load the image when necessary (when we get the image)
        image = Image.open(image_path).convert('RGB')
        image = self._apply_transforms(image)
        label = self.labels[id]
        id_for_label = self.id_label[label]
        # skorch expects a tuple for their fit method
        return (image,int(id_for_label))
    
    def _apply_transforms(self,image):
        return self.transforms(image)

class CNN(nn.Module):

    def __init__(self, n_conv_layers=3, starting_filters=16, n_labels=3, kernel_size=3):
        super().__init__()
        conv_layers = []
        in_channels = 3 
        out_channels = starting_filters
        actual_size = 224

        for _ in range(n_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2,2))
            in_channels = out_channels
            out_channels *= 2
            actual_size //= 2

        self.features = nn.Sequential(*conv_layers, nn.AdaptiveAvgPool2d((1, 1)))

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_labels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classification(x)
        return x

""" Test model performance on the dataset as if the data scientist was
allowed to see the information. This will let us compare accuracy against
privacy.
"""
def main():

    images = []
    labels = []
    # Load the image paths: 80% train 20% test
    for label, image_list in load_images(DATA_PATH).items():
        images.extend(image_list)
        labels.extend([label]*len(image_list))

    X_train, X_test, y_train, y_test = train_test_split(
        images, 
        labels,
        test_size=0.20, 
        stratify=labels, 
        random_state=42
    )

    unique_labels = sorted(set(y_train))
    label_hash = {label: i for i, label in enumerate(unique_labels)}

    # Transform the paths into images, apply transforms and create tensors
    train_ds = ImageDataset(X_train, y_train, label_hash)
    test_ds = ImageDataset(X_test, y_test, label_hash)

    # Create the NeuralNetClassifier
    model = NeuralNetClassifier(
        module=CNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        batch_size=16,
        max_epochs=10,
        iterator_train__num_workers=8
    )

    # Choose the parameters to be optimized in the GridSearch
    params = {
        # CNN parameters
        'module__n_conv_layers': [3, 4],
        'module__starting_filters': [8, 16],
        #'module__kernel_size': [3, 5],
        # Optimization parameters
        #'lr': [0.001, 0.01]
        #'optimizer__momentum': [0.9]
    }

    X_train = SliceDataset(train_ds, idx=0)
    y_train = SliceDataset(train_ds, idx=1)
    X_test = SliceDataset(test_ds, idx=0)
    y_test = SliceDataset(test_ds, idx=1)

    y_true = [y for y in y_test]

    grid_search = GridSearchCV(model, params, refit=True, cv=3)
    grid_search.fit(X_train,y_train)
    
    print(f"Best score: {grid_search.best_score_}")
    print(f"Best parameters: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Confusion matrix: {confusion_matrix(y_true, y_pred)}")
    print(f"Report: {classification_report(y_true, y_pred)}")


if __name__ == "__main__":
    main()