import torch
import time
import gc
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, Array
from flwr.clientapp import ClientApp

from src.task import load_data
from src.task import test as test_fn
from src.task import train_fedavg as train_fn_avg
from src.task import train_scaffold as train_fn_scaffold
from src.task import train_fedprox as train_fn_prox
from src.dp_utils import add_differential_privacy_to_updates
from utils.config import load_config
from src.model import MODEL, CNN_Local

TRAIN_FN = {
    "fedavg": lambda extra, common: train_fn_avg(
        **common
    ),
    "fedavg-dp": lambda extra, common: train_fn_avg(
        **common
    ),
    "fedavg_precision_based": lambda extra, common: train_fn_avg(
        **common
    ),
    "scaffold": lambda extra, common: train_fn_scaffold(
        **extra, **common
    ),
    "fedprox": lambda extra, common: train_fn_prox(
        **extra, **common
    ),
    "malicious_actor_detector": lambda extra, common: train_fn_avg(
        **common
    ),
    "malicious_server_existing_inference": lambda extra, common: train_fn_avg(
        **common
    ),
    "malicious_server_metric_based": lambda extra, common: train_fn_avg(
        **common
    )
}

# Initialize the client application
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):

    config = load_config("./data/processed/config.json")
    dataset = config["dataset"]

    strat = context.run_config.get("strategy","fedavg")

    if strat == "scaffold" and "c_local" not in context.state.array_records:
        context.state.array_records["c_local"] = ArrayRecord(
            {k: Array(np.zeros_like(v.numpy())) for k, v in msg.content["arrays"].items()})

    # Load the CNN
    model = MODEL.get(dataset,CNN_Local)()
    # Receive the global aggregated weights
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Move to gpu if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the local data
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]

    trainloader, _ = load_data(partition_id, batch_size)

    extra = {
        # SCAFFOLD:
        "global_c": msg.content.get("global-control",0),
        "local_c": context.state.array_records.get("c_local",0),
        # FedProx:
        "proximal_mu": msg.content["config"].get("proximal-mu",0),
        "inexact_threshold": context.run_config.get("inexact-threshold",0)
    }

    common_parameters = {
        "model": model,
        "trainloader": trainloader,
        "epochs": context.run_config["local-epochs"],
        "lr": msg.content["config"]["lr"],
        "device": device
    }

    # Call the training function
    start_time = time.perf_counter()
    train_fn= TRAIN_FN.get(strat,TRAIN_FN["fedavg"])
    train_loss, train_acc, w_diff, c_diff, new_c = train_fn(extra=extra,common=common_parameters)
    end_time = time.perf_counter()
    train_duration = end_time - start_time
    print(train_duration)

    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "num-examples": len(trainloader.dataset),
    }

    metric_record = MetricRecord(metrics)

    # Update the local c parameter
    if strat == "scaffold":
        context.state.array_records["c_local"] = new_c
        model_record = ArrayRecord(torch_state_dict=w_diff)
        c_record = ArrayRecord(torch_state_dict=c_diff)
        content = RecordDict({"arrays": model_record, "c_values": c_record, "metrics": metric_record})
    elif strat == "malicious_actor_detector" and partition_id == 0:
        #model_record = ArrayRecord(aggregate_malicious_vector(model.state_dict()))
        # for now use CNN().state_dict() which generates random parameters to check
        # if Krum can detect this in the server
        print("The malicious actor has been chosen for training")
        model_record = ArrayRecord(MODEL.get(dataset,"local")().state_dict()) #random vector of parameters
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
    elif strat == "fedavg-dp":
        torch.manual_seed(config["seed"]) # for gaussian noise 
        # In the case of differential privacy we first calculate the updates (difference between the global
        # model and the model after being trained) and then return global_weights + noisy_updates
        global_weights = msg.content["arrays"].to_torch_state_dict()
        local_weights = model.state_dict()
        updates = {key: local_weights[key] - global_weights[key] for key in global_weights.keys()}
        trainable_keys = [key for key in updates.keys() if 'weight' in key or 'bias' in key]
        trainable_values = [updates[tkey] for tkey in trainable_keys]

        dp_updates = add_differential_privacy_to_updates(
            updates=trainable_values,
            clipping=msg.content["config"]["clipping"],
            epsilon=msg.content["config"]["epsilon"],
            delta=msg.content["config"]["delta"],
            device=device
        )

        weights_with_noise = {}
        dp_updates_map = dict(zip(trainable_keys, dp_updates))
        for key in updates.keys():
            if key in dp_updates_map:
                weights_with_noise[key] = global_weights[key] + dp_updates_map[key]
            else:
                # this is to prevent adding noise to layer params such as batchnorms running_mean
                weights_with_noise[key] = local_weights[key]

        model_record = ArrayRecord(weights_with_noise)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
    else: # FedAvg & FedProx
        for key, value in model.state_dict().items():
            print(f"Key: {key} | Shape: {str(value.shape)} | Count: {value.numel()} | Type: {value.dtype}")
        model_record = ArrayRecord(model.state_dict())
        content = RecordDict({"arrays": model_record, "metrics": metric_record})

    # Collect garbage
    del model
    del trainloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):

    config = load_config("./data/processed/config.json")
    mappings = load_config("./data/processed/label_mappings.json")
    dataset = config["dataset"]

    strat = context.run_config.get("strategy","fedavg")
    # Load the model
    model = MODEL.get(dataset,"local")()
    # Get the aggregated globla weights
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Move to GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]

    _, valloader = load_data(partition_id, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc, confusion_matrix, precision = test_fn(
        model,
        valloader,
        device,
        [int(label) for label in mappings["id_to_label"].keys()],
    )

    # Construct and return reply Message with metrics after evaluation
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "confusion_matrix": confusion_matrix,
        "num-examples": len(valloader.dataset),
        "precision": float(precision)
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    # Collect garbage
    del model
    del valloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)
