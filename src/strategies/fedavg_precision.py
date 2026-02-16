from flwr.serverapp.strategy import FedAvg
from collections.abc import Iterable
from flwr.serverapp.strategy.strategy_utils import sample_nodes, validate_message_reply_consistency, aggregate_arrayrecords
from logging import INFO
from flwr.app import MessageType
from flwr.common import (
    Message,
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    NDArray,
    RecordDict,
    log,
)
from typing import cast
import numpy as np

"""
Implementation of a custom strategy that attempts to handle
unreliable/low-quality clients. Rewards high-performing nodes 
during evaluation increasing their weight during aggregation.

"""
class PrecisionWeightedFedAvg(FedAvg):
    
    def __init__(self,
                *args,                         # FedAvg parameters
                **kwargs):                     # FedAvg parameters
        super().__init__(*args,**kwargs)
        
        self.client_precisions = {}            # Store evaluation metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        if valid_replies:
            for msg in valid_replies:
                if "metrics" in msg.content:
                    metrics = msg.content["metrics"]
                    if "precision" in metrics:
                        client_id = msg.metadata.src_node_id
                        print(metrics["precision"])
                        self.client_precisions[client_id] = metrics["precision"]

        return super().aggregate_evaluate(server_round, replies)

    def aggregate_train(self, server_round, replies):
        # If there are still no client precisions (no evaluation round has ocurred)
        # aggregate train by number of examples (normal FedAvg)
        if not self.client_precisions:
            return super().aggregate_train(server_round, replies)
        else:
            return self._aggregate_train_precision(server_round, replies)
        
    def _aggregate_train_precision(self, server_round, replies):
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            # inject the precision of each client in each
            # client's message content
            for msg in valid_replies:
                client_id = msg.metadata.src_node_id
                weight = self.client_precisions[client_id]
                msg.content["precision_eval"] = ConfigRecord({"val": float(weight)}) # TODO
                print(weight)

            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                reply_contents,
                weighting_metric_name="precision_eval"
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return arrays, metrics
    
    def summary(self):
        print("Precision-weighted FedAvg.")