import io
import time
from typing import cast
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image

import torch
import torch.nn.functional as F
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
import numpy as np
from flwr.common import log, logger, NDArray, Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info
from src.task import load_data, CNN
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

# Source inference attack: could this information have been used by this specific client?
""" 
We assume an IID setting for this experiment (consider this when generating the data)
We generate an existing inference attack, answering the following question:
"Was this type of information used in the training?"

OBJECTIVE: 
    to perform a global membership inference attack to identify the
    underlying data domain used in the Federated Learning setting.

MECHANISM: 
    the server prepares two anchor records:
    - Target: an image from the actual dataset (/data/raw)
    - Null: a totally unrelated image (/data/experimental)

    The setting will perform a white-box attack where the server
    wants to infer whether a certain image was used in training.

GOAL: 
    to prove that even without access to the original dataset, the server
    can identify the type of data the model is being trained with. 

    This experiment aims to represent the importance of Differential Privacy
    in a Federated Learning setting to ensure privacy.
"""
class MaliciousServerMetricBasedAttack(FedAvg):

    transforms = Compose([Resize((224,224)), ToTensor(), 
                                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __init__(self, target_img, null_img, num_rounds, *args, **kwargs):
        # Load the images that are going to be used
        target = Image.open(target_img).convert("RGB")
        null = Image.open(null_img).convert("RGB")
        self.target_img = self._apply_transforms(target) # Seen by the model
        self.null_img = self._apply_transforms(null) # Never seen by the model
        self.rounds = num_rounds
        # Remember last round model parameters to compute the gradients
        super().__init__(*args, **kwargs)

    def _apply_transforms(self, image):
        tensor = self.transforms(image)
        return tensor.unsqueeze(0)
        
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        arrays, metrics = super().aggregate_train(server_round,replies)
        # If its the last aggregation round
        #if self.rounds == server_round:
        model = CNN()
        model.load_state_dict(arrays.to_torch_state_dict())

        model.eval()
        target_label = torch.argmax(model(self.target_img), dim=1)
        null_label = torch.argmax(model(self.null_img), dim=1)

        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        model.zero_grad()
        loss = criterion(model(self.target_img), target_label)
        loss.backward()
        target_norm = sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)

        model.zero_grad()
        loss = criterion(model(self.null_img), null_label)
        loss.backward()
        null_norm = sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)

        print(f"Target norm: {target_norm}")
        print(f"Null norm: {null_norm}")

        return arrays, metrics
    