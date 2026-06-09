from logging import INFO
from typing import Iterable

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import FedAvg


# Unimplemented
class AdaptiveFL(FedAvg):
    def __init__(self, *args, **kwargs):
        
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
            
