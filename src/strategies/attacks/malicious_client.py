import io
import time
from typing import cast
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
import numpy as np
from flwr.common import log, logger, NDArray, Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info


class MaliciousActorIIDDetector(FedAvg):
        
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

        # Malicious actor detection: euclidean distance using Krum
        client_vectors = []

        # for each client
        for record in records:
            parameters = []
            # for each type of parameter received (weights, bias...)
            for record_item in record.array_records.values():
                # name and value of the parameter
                for key, value in record_item.items():
                    # transform into a 1D vector  
                    parameters.append(value.numpy().flatten())

            client_vectors.append(np.concatenate(parameters))

        client_scores = []

        # calculate all client scores
        for i in range(0,len(client_vectors)):
            actual_client = client_vectors[i]
            client_distances = []
            for j in range(0,len(client_vectors)):
                if i == j:
                    continue
                compared_client = client_vectors[j]
                client_distances.append(np.linalg.norm(actual_client - compared_client))
            # assume only one malicious client and six chosen clients for training
            # n - f - 2 = 6 - 1 - 2 = choose the 3 nearest neighbours
            n = 6
            f = 1
            client_distances.sort()
            num_neighbors = n - f - 2
            client_scores.append(sum(client_distances[:num_neighbors])) 

        print(f"Client scores: {client_scores}")

        suspicious_score = max(client_scores)
        untrusted_client = client_scores.index(suspicious_score)

        aggregated_np_arrays: dict[str, NDArray] = {}

        for record, weight in zip(records, weight_factors, strict=True):
            client_id = 0
            for record_item in record.array_records.values():
                # aggregate in-place
                if client_id == untrusted_client:
                    continue
                for key, value in record_item.items():
                    if key not in aggregated_np_arrays:
                        aggregated_np_arrays[key] = value.numpy() * weight
                    else:
                        aggregated_np_arrays[key] += value.numpy() * weight
            client_id += 1

        return ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()}
        )