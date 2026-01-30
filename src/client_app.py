import torch
import gc
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, Array
from flwr.clientapp import ClientApp

from src.task import CNN, load_data
from src.task import test_fedavg as test_fn
from src.task import train_fedavg as train_fn_avg
from src.task import train_scaffold as train_fn_scaffold

TRAIN_FN = {
    "fedavg": lambda _, common: train_fn_avg(
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

    strat = context.run_config.get("strategy","fedavg")

    if strat == "scaffold" and "c_local" not in context.state.array_records:
        context.state.array_records["c_local"] = ArrayRecord(
            {k: Array(np.zeros_like(v.numpy())) for k, v in msg.content["arrays"].items()})

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
        "global_c": msg.content["global-control"],
        "local_c": context.state.array_records["c_local"]
    }

    common_parameters = {
        "model": model,
        "trainloader": trainloader,
        "epochs": context.run_config["local-epochs"],
        "lr": msg.content["config"]["lr"],
        "device": device
    }

    # Call the training function
    train_fn= TRAIN_FN.get(strat,TRAIN_FN["fedavg"])
    train_loss, w_diff, c_diff, new_c = train_fn(extra=extra,common=common_parameters)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }

    metric_record = MetricRecord(metrics)

    # Update the local c parameter
    if strat == "scaffold":
        context.state.array_records["c_local"] = new_c
        model_record = ArrayRecord(torch_state_dict=w_diff)
        c_record = ArrayRecord(torch_state_dict=c_diff)
        content = RecordDict({"arrays": model_record, "c_values": c_record, "metrics": metric_record})
    else:
        model_record = ArrayRecord(torch_state_dict=w_diff)
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
