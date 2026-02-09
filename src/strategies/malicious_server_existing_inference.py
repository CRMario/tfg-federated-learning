import io
import time
from typing import cast
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image

import torch
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

    The server will compute the gradient norm of these two images
    against the received client updates.

GOAL: 
    to prove that even without access to the original dataset, the server
    can identify the type of data the model is being trained with. 
    Consider that it will not figure out if that specific image was used for
    training. However, the server should not know anything about the dataset,
    yet it can figure out the general type of data being used.

    This experiment aims to represent the importance of Differential Privacy
    in a Federated Learning setting to ensure privacy.
"""
class MaliciousServerExistingInferenceAttack(FedAvg):

    transforms = Compose([Resize((224,224)), ToTensor(), 
                                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __init__(self, target_img, null_img, lr, *args, **kwargs):
        # Load the images that are going to be used
        target = Image.open(target_img).convert("RGB")
        null = Image.open(null_img).convert("RGB")
        self.target_img = self._apply_transforms(target)
        self.null_img = self._apply_transforms(null)
        # Remember last round model parameters to compute the gradients
        self.last_round_model_parameters = None
        self.lr = lr
        super().__init__(*args, **kwargs)

    def _apply_transforms(self, image):
        return self.transforms(image)

    def configure_train(self, server_round, arrays, config, grid):
        # Before every training save last round global model parameters
        self.last_round_model_parameters = {k: v for k, v in arrays.items()}
        return super().configure_train(server_round, arrays, config, grid)
        
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = self.aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return arrays, metrics
    
    def aggregate_arrayrecords(self, records, weighting_metric_name):
        """Perform weighted aggregation on all ArrayRecords. Detect if there
        is a malicious actor."""
        weights: list[float] = []
        for record in records:
            # Get the first (and only) MetricRecord in the record
            metricrecord = next(iter(record.metric_records.values()))
            # Because replies have been checked for consistency,
            # we can safely cast the weighting factor to float
            w = cast(float, metricrecord[weighting_metric_name])
            weights.append(w)

        # Average
        total_weight = sum(weights)
        weight_factors = [w / total_weight for w in weights]

        # Before aggregating, create a Shadow Model to compute the attack
        state_dict = {k: torch.from_numpy(v.numpy()) for k, v in self.last_round_model_parameters.items()}
        shadow_model = CNN()
        shadow_model.load_state_dict(state_dict)
        # We obtain the gradients ("directions") the model would take
        # after being trained with each of these images
        target_grad = self.compute_shadow_grad(shadow_model, self.target_img, 1, self.lr)
        null_grad = self.compute_shadow_grad(shadow_model, self.null_img, 1, self.lr)

        trainable_keys = {key for key, param in shadow_model.named_parameters() if param.requires_grad}

        client_grad_vectors = []

        for record in records:
            parameters = []
            # for each type of parameter received (weights, bias...)
            for record_item in record.array_records.values():
                # name and value of the parameter
                for key, value in record_item.items():
                    if key in trainable_keys:
                        # transform into a 1D vector
                        grad_value = torch.from_numpy(value.numpy()) - torch.from_numpy(self.last_round_model_parameters[key].numpy())
                        parameters.append(grad_value.flatten())

            client_grad_vectors.append(torch.cat(parameters))

        similarities_target = []
        similarities_null = []

        for client_grad in client_grad_vectors:
            similarity_with_target = torch.nn.functional.cosine_similarity(client_grad,target_grad,dim=0)
            similarity_with_null = torch.nn.functional.cosine_similarity(client_grad,null_grad,dim=0)
            similarities_target.append(similarity_with_target)
            similarities_null.append(similarity_with_null)
            print(f"Similarity with target: {similarity_with_target}")
            print(f"Similarity with null: {similarity_with_null}")
            
        print(f"Mean similarity with target: {sum(similarities_target)/len(similarities_target)}")
        print(f"Mean similarity with null: {sum(similarities_null)/len(similarities_null)}")

        aggregated_np_arrays: dict[str, NDArray] = {}

        for record, weight in zip(records, weight_factors, strict=True):
            for record_item in record.array_records.values():
                for key, value in record_item.items():
                    if key not in aggregated_np_arrays:
                        aggregated_np_arrays[key] = value.numpy() * weight
                    else:
                        aggregated_np_arrays[key] += value.numpy() * weight

        return ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()}
        )
    
    def compute_shadow_grad(self, model, image, label, lr):
        target = torch.tensor([label])
        
        if image.dim() == 3:
            image = image.unsqueeze(0)

        model.zero_grad()
        output = model(image)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        grads = torch.cat([p.grad.view(-1) for _, p in model.named_parameters() if p.grad is not None])

        shadow_update = -lr * grads
        return shadow_update