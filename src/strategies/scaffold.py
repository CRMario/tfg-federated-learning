from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from logging import INFO
from flwr.app import MessageType
from flwr.common import (
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

class SCAFFOLD(FedAvg):
    """
    SCAFFOLD Federated Learning Strategy.

    Implementation based on https://arxiv.org/abs/1910.06378
    """
    def __init__(self,
                global_lr,                     # Specify the global learning rate
                initial_model_parameters,      # SCAFFOLD must remember last round's parameters
                *args,                         # FedAvg parameters
                **kwargs):                     # FedAvg parameters
        super().__init__(*args,**kwargs)
        # Initialize the global control variable (as many ceros as parameters)
        # Serialize them too since they will be sent to the clients
        initial_c_dict = {k: Array(ndarray=np.zeros_like(v.numpy()))
                          for k,v in initial_model_parameters.items()}
        self.c_global = ArrayRecord(initial_c_dict)
        # Set the global learning rate
        self.global_lr = global_lr
        # Save the previous round parameters
        self.last_round_model_parameters = initial_model_parameters
        self.num_clients = self.min_available_nodes

    def configure_train(self, server_round, arrays, config, grid):
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, 
             "global-control": self.c_global,
             self.configrecord_key: config}
        )

        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self.__check_and_log_replies(replies,is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
        
            arrays_diff, c_diff = self._aggregate_clients_diff(reply_contents)
            arrays, c = self._aggregate(arrays_diff,c_diff)
            metrics = self.train_metrics_aggr_fn(reply_contents)
            self.c_global = c
            self.last_round_model_parameters = arrays

        return arrays, metrics

    def configure_evaluate(self, server_round, arrays, config, grid):
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round, replies):
        return super().aggregate_evaluate(server_round, replies)

    def summary(self):
        print("SCAFFOLD Strategy implementation from scratch.")

    def _aggregate_clients_diff(self, records):
        """Perform aggregation from the clients models diff"""

        m = len(records)
        weight = 1 / m

        aggregated_np_arrays: dict[str, NDArray] = {}
        aggregated_c_values: dict[str, NDArray] = {}

        for record in records: #each package sent by the client

            # get the parameters and local control variables
            client_w_diff = record.array_records["arrays"]
            client_c_diff = record.array_records["c_values"]

            # aggregate parameters
            for key, value in client_w_diff.items():
                if key not in aggregated_np_arrays:
                    aggregated_np_arrays[key] = value.numpy() * weight
                else:
                    aggregated_np_arrays[key] += value.numpy() * weight

            # aggregate control variables
            for key, value in client_c_diff.items():
                if key not in aggregated_c_values:
                    aggregated_c_values[key] = value.numpy() * weight
                else:
                    aggregated_c_values[key] += value.numpy() * weight

        return ArrayRecord(
            {k: Array(ndarray=v) for k, v in aggregated_np_arrays.items()}
        ), ArrayRecord(
            {k: Array(ndarray=v) for k, v in aggregated_c_values.items()}
        )
    
    def _aggregate(self,arrays,c_values):

        aggregated_np_array:  dict[str, NDArray] = {}
        aggregated_c_values:  dict[str, NDArray] = {}

        for key, value in arrays.items():
            if key not in aggregated_np_array:
                aggregated_np_array[key] = (value.numpy() * self.global_lr) + self.last_round_model_parameters[key].numpy()
            else:
                aggregated_np_array[key] += (value.numpy() * self.global_lr) + self.last_round_model_parameters[key].numpy()
        
        weight = len(c_values) / self.num_clients

        for key, value in c_values.items():
            if key not in aggregated_c_values:
                aggregated_c_values[key] = (value.numpy() * weight) + self.c_global[key].numpy()
            else:
                aggregated_c_values[key] += (value.numpy() * weight) + self.c_global[key].numpy()
        
        return ArrayRecord(
            {k: Array(ndarray=v) for k, v in aggregated_np_array.items()}
        ), ArrayRecord(
            {k: Array(ndarray=v) for k, v in aggregated_c_values.items()}
        )
        