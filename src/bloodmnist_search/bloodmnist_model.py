import torch
import torch.nn as nn
import numpy as np
from medmnist import BloodMNIST
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV

class CNN(nn.Module):

    def __init__(self, n_conv_layers=3, n_fc_layers=2, starting_filters=16, n_labels=8, kernel_size=3):
        super().__init__()
        conv_layers = []
        in_channels = 3 
        out_channels = starting_filters
        actual_size = 28

        for i in range(n_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            if i < n_conv_layers - 1: 
                conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2
            actual_size //= 2

        self.features = nn.Sequential(*conv_layers, nn.AdaptiveAvgPool2d((2, 2)))

        fc_layers = [nn.Flatten()]

        current_dim = in_channels * 4
        hidden_dim = 128

        for _ in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(current_dim, hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(0.5))
            current_dim = hidden_dim
        fc_layers.append(nn.Linear(current_dim, n_labels))

        self.classification = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classification(x)
        return x

def main():

    train_set = BloodMNIST(split='train', root='./data/bloodmnist', download=True)
    validation_set = BloodMNIST(split='val', root='./data/bloodmnist', download=True)

    # Create the NeuralNetClassifier
    model = NeuralNetClassifier(
        module=CNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        batch_size=128,
        max_epochs=50,
        iterator_train__num_workers=8
    )

    # Choose the parameters to be optimized in the GridSearch
    params = {
        # CNN parameters
        'module__n_conv_layers': [2, 3],
        'module__starting_filters': [16, 32],
        'module__n_fc_layers': [2,3],
        # Other parameters
        'batch_size': [64, 128]
    }

    # Match X_train and y_train to the format scikit-learn expects
    X_train = np.concatenate([train_set.imgs, validation_set.imgs], axis=0).transpose(0, 3, 1, 2)
    y_train = np.concatenate([train_set.labels, validation_set.labels], axis=0).flatten()

    # scikit-learn does not use transformers so normalise X_train manually to [0,1]
    X_train = X_train.astype('float32') / 255.0

    grid_search = RandomizedSearchCV(model, params, refit=True, n_iter=10, cv=3, scoring="accuracy", verbose=2)
    grid_search.fit(X_train,y_train)
    
    print(f"Best score: {grid_search.best_score_}")
    print(f"Best parameters: {grid_search.best_params_}")

if __name__ == "__main__":
    main()