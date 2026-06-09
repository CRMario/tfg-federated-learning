from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes, validate_message_reply_consistency
from logging import INFO
from flwr.app import MessageType
from flwr.common import (
    Message,
    Array,
    ArrayRecord,
    NDArray,
    RecordDict,
    log,
)
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
        valid_replies, _ = self._check_and_log_replies(replies,is_train=False)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
        
            arrays_diff, c_diff = self._aggregate_clients_diff(reply_contents)
            arrays, c = self._aggregate(arrays_diff,c_diff)
            metrics = self.train_metrics_aggr_fn(reply_contents,self.weighted_by_key)
            self.c_global = c
            self.last_round_model_parameters = arrays

        return arrays, metrics

    def configure_evaluate(self, server_round, arrays, config, grid):
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round, replies):
        return super().aggregate_evaluate(server_round, replies)

    def summary(self):
        print("SCAFFOLD Strategy implementation using Flower framework.")

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

        return aggregated_np_arrays, aggregated_c_values
        
    
    def _aggregate(self,arrays,c_values):

        aggregated_np_array:  dict[str, NDArray] = {}
        aggregated_c_values:  dict[str, NDArray] = {}

        for key, value in arrays.items():
            aggregated_np_array[key] = (value * self.global_lr) + self.last_round_model_parameters[key].numpy()
        
        weight = len(c_values) / self.num_clients

        for key, value in c_values.items():
            aggregated_c_values[key] = (value * weight) + self.c_global[key].numpy()
        
        return ArrayRecord(
            array_dict={k: Array(np.asarray(v)) for k, v in aggregated_np_array.items()}
        ), ArrayRecord(
            array_dict={k: Array(np.asarray(v)) for k, v in aggregated_c_values.items()}
        )

    def _check_and_log_replies(self, replies, is_train, validate = True):
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )

        # Log errors
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        # Ensure expected ArrayRecords and MetricRecords are received
        if validate and valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=False, # We do not validate ArrayRecords
            )

        return valid_replies, error_replies