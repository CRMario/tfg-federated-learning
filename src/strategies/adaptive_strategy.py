import io
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info


# Unimplemented
class AdaptiveFL(FedAvg):
    def __init__(self, *args, **kwargs):

        # Keep in memory the algorithm that is currently being used
        self.current_alg = "FedAvg"
        
        # FedProx parameters
        self.mu = 0.0

        # Bulyan parameters
        self.f = 3 #temporal
        self.client_reliability = {}
        self.client_history = {}

        # Switch triggers
        # If drift > drift_threshold: change mu depending on similarity
        self.drift_threshold = 0.5
        
        self.anomaly_threshold = 1.5

        # Initialize the FedAvg parameters
        super().__init__(*args, **kwargs)

    def aggregate_train(
        server_round: int, replies: Iterable[Message]
    )-> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate the learned parameters from the clients obtained
        in the last round of federated learning"""

        # Normal FedAvg aggregation
        if server_round % 5 != 0:
            super().aggregate_train(server_round,replies)

        # Every five rounds check the replies sent by the clients.

        for reply in replies:
            reply.content
            
