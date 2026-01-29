import torch
import gc
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.task import CNN, load_data
from src.task import test_fedavg as test_fn
from src.task import train_fedavg as train_fn_avg
from src.task import test_scaffold as test_fn_scaffold
from src.task import train_scaffold as train_fn_scaffold

TRAIN_FN = {
    "fedavg": lambda extra, common: train_fn_avg(
        **common
    ),
    "scaffold": lambda extra, common: train_fn_scaffold(
        **extra, **common
    )
}

# Initialize the client application
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):

    if "c_local" not in context.node_config:
        context.node_state["c_local"] = {k: np.zeros_like(v) for k, v in msg.content["arrays"].items()}

    # Load the CNN
    model = CNN()
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
        "c_global": msg.content["global-control"],
        "c_local": context.node_state["c_local"]
    }

    common_parameters = {
        "model": model,
        "trainloader": trainloader,
        "local_epochs": context.run_config["local-epochs"],
        "lr": msg.content["config"]["lr"],
        "device": device
    }

    # Call the training function
    strat = context.run_config.get("strategy","fedavg")
    train_fn= TRAIN_FN.get(strat,TRAIN_FN["fedavg"])
    train_loss = train_fn(extra=extra,common=common_parameters)

    # TODO: modify train function for SCAFFOLD, return new c, etc.

    # Return a reply with the metrics after training
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
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

    # Load the model
    model = CNN()
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
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message with metrics after evaluation
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
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
