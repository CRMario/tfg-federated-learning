from src.config.constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rand


"""
This will all change it's an initial prototype
"""

class FedAvgServer:

    alg_name = "FedAvg"
    return_diff = False

    def __init__(self,n_labels,clients,rounds):
        self.n_labels = n_labels
        self.clients = clients
        self.K = len(clients)
        self.C = 1
        self.rounds = rounds
        self.model = CNNModel(n_labels) # Initialize a global model
        self.client_models = []
        self.local_epochs = 10
        self.local_batch_size = 5

    def choose_clients(self):
        m = max(int(self.C*self.K),1)
        return rand.sample(self.clients,m)

    def aggregate_client_models(self,client_models_states,data_sizes):
        m = sum(data_sizes)
        # Initialize the new global parameters
        new_state_dict = {}
        previous_global_model = self.model.state_dict()
        for key in previous_global_model:
            new_state_dict[key] = torch.zeros_like(previous_global_model[key])

            for client in enumerate(client_models_states):
                weight = data_sizes[client] / m
                client_params = client_models_states[client]
                new_state_dict[key] += client_params[key] * weight

        return new_state_dict

    def update_global_model(self,client_models_states,data_sizes):
        new_parameters = self.aggregate_client_models(client_models_states,data_sizes)
        self.model.load_state_dict(new_parameters)
        
    def run_round(self):
        client_models_states = []
        data_sizes = []

        selected_clients = self.choose_clients()
        for client in selected_clients:

            update, size = client.local_train(self.model.state_dict())
            client_models_states.append(update)
            data_sizes.append(size)

    def train(self):
        pass

    def run(self):
        pass

class CNNModel(nn.Module):
    def __init__(self,n_labels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # RGB channels input
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_labels) # Output three classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x