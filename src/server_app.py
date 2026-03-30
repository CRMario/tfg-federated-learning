import torch
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as ppt
import os
import mlflow
import time
import mlflow.pytorch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.serverapp.strategy.fedavg import FedAvg
from src.strategies.scaffold import SCAFFOLD
from src.strategies.attacks.malicious_client import MaliciousActorIIDDetector
from src.strategies.fedavg_precision import PrecisionWeightedFedAvg
from utils.utils import aggregate_metricrecords
from utils.config import load_config
from src.task import global_evaluate

from src.model import MODEL, CNN_Local

STRATEGY = {
    "fedavg": lambda configuration, initial_params, common: FedAvg(
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common
    ),
    "fedavg-dp": lambda configuration, initial_params, common: FedAvg(
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common
    ),
    "fedavg_precision_based": lambda configuration, initial_params, common: PrecisionWeightedFedAvg(
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common
    ),
    "scaffold": lambda configuration, initial_params, common: SCAFFOLD(
        global_lr=configuration.get("global-lr",1.0),
        initial_model_parameters=initial_params,
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common
    ),
    "fedprox": lambda configuration, initial_params, common: FedProx(
        proximal_mu=configuration.get("proximal-mu",1.0),
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common
    ),
    "malicious_actor_detector": lambda configuration, initial_params, common: MaliciousActorIIDDetector(
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common
    ),
}

# Initialize the server application
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # Set up MLFlow tracking uri
    mlflow.set_tracking_uri("http://localhost:5000")

    # Read run configuration
    config = context.run_config

    # Get the strategy from the configuration
    strat = config.get("strategy", "fedavg") #default to fedavg

    # Set up MLFlow experiment
    mlflow.set_experiment(strat)

    current_time = time.strftime("%Y%m%d-%H%M")
    current_run_name = f"{strat}_{current_time}"

    with mlflow.start_run(run_name=current_run_name):

        partition_config = load_config("./data/processed/config.json")

        mlflow.log_params(config)
        mlflow.log_params(partition_config)
        mlflow.log_param("strategy", strat)
        mlflow.log_artifact("data/processed/config.json", artifact_path="configs")
        mlflow.log_artifact("data/processed/label_mappings.json", artifact_path="mappings")

        # Read the parameters common to all strategies
        fraction_evaluate: float = context.run_config["fraction-evaluate"]
        fraction_train: float = context.run_config["fraction-train"]
        num_rounds: int = context.run_config["num-server-rounds"]
        lr: float = context.run_config["learning-rate"]

        common_params = {
            "fraction_evaluate": fraction_evaluate,
            "fraction_train": fraction_train,
            "min_train_nodes": 6, 
            "min_evaluate_nodes": 8,  
            "min_available_nodes": 8, 
        }

        # Load global model
        dataset = partition_config["dataset"]
        global_model = MODEL.get(dataset,CNN_Local)()
        # Get the initial weights. They are randomly initialized.
        arrays = ArrayRecord(global_model.state_dict())

        # Initialize the strategy
        strategy_builder = STRATEGY.get(strat,STRATEGY["fedavg"]) #default to fedavg
        strategy = strategy_builder(configuration=config,initial_params=arrays,common=common_params)

        # Begin the simulation with the selected strategy
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate
        )

        for round, metrics in result.evaluate_metrics_clientapp.items():
            for metric, value in metrics.items():
                mlflow.log_metric(f"eval_{metric}", value, step=round)
        for round, metrics in result.train_metrics_clientapp.items():
            for metric, value in metrics.items():
                mlflow.log_metric(f"train_{metric}", value, step=round)
    
        plot_results(result)
        mlflow.log_artifact("outputs/global_metrics.png")
        mlflow.log_artifact("outputs/label_metrics.png")

        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        global_model.load_state_dict(state_dict)
        mlflow.pytorch.log_model(global_model, name=f"final_model_{strat}")

def plot_results(result):
    #result[round][metric_key]
    rounds = sorted(result.evaluate_metrics_clientapp.keys())

    f1_per_label = {} #example: {'lung cancer': [0.8, 0.9, 0.91]...}
    precision_per_label = {}
    recall_per_label = {}
    eval_accuracy, eval_loss, train_accuracy, train_loss = [], [], [], []

    for round in rounds:
        evaluation_metrics = result.evaluate_metrics_clientapp[round]
        for key, value in evaluation_metrics.items():

            if key.startswith("f1_score_label"):
                label_id = key.replace("f1_score_label", "")
                if label_id not in f1_per_label.keys():
                    f1_per_label[label_id] = [value]
                else:
                    f1_per_label[label_id].append(value)

            elif key.startswith("precision_label"):
                label_id = key.replace("precision_label", "")
                if label_id not in precision_per_label.keys():
                    precision_per_label[label_id] = [value]
                else:
                    precision_per_label[label_id].append(value)

            elif key.startswith("recall_label"):
                label_id = key.replace("recall_label", "")
                if label_id not in recall_per_label.keys():
                    recall_per_label[label_id] = [value]
                else:
                    recall_per_label[label_id].append(value)

            elif key == "eval_acc":
                eval_accuracy.append(value)

            elif key == "eval_loss":
                eval_loss.append(value)

    for round in rounds:
        train_metrics = result.train_metrics_clientapp[round]
        for key, value in train_metrics.items():

            if key == "train_acc":
                train_accuracy.append(value)

            elif key == "train_loss":
                train_loss.append(value)

    # Plotting global results: evaluation and train loss and accuracy
    figure, (ax1, ax2) = ppt.subplots(2,1,figsize=(12,10))
    ax1.plot(rounds, eval_accuracy, label="Evaluation accuracy", color='blue')
    ax1.plot(rounds, train_accuracy, label="Train accuracy", color='red')
    ax1.set_title("Aggregated accuracy over rounds", fontsize=14)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(rounds, eval_loss, label="Evaluation loss", color='blue')
    ax2.plot(rounds, train_loss, label="Train loss", color='red')
    ax2.set_title("Aggregated loss over rounds", fontsize=14)
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ppt.tight_layout()
    metric_output_dir = 'outputs'
    if not os.path.exists(metric_output_dir):
        os.makedirs(metric_output_dir)
        
    save_path = os.path.join(metric_output_dir, "global_metrics.png")

    ppt.savefig(save_path)
    print(f"Global metrics image saved as: {save_path}")
    ppt.close(figure)

    # Plotting aggregated metrics: precision, recall, f1-score

    sorted_labels = []
    for label in f1_per_label.keys():
        sorted_labels.append(int(label))
    sorted_labels.sort()

    figure, (ax1, ax2, ax3) = ppt.subplots(1,3,figsize=(30,6))
    for label in sorted_labels:
        ax1.plot(rounds, precision_per_label[str(label)], label=f"Label {label}")
        ax2.plot(rounds, recall_per_label[str(label)], label=f"Label {label}")
        ax3.plot(rounds, f1_per_label[str(label)], label=f"Label {label}")
    
    ax1.set_title("Precision per label over rounds", fontsize=14)
    ax2.set_title("Recall per label over rounds", fontsize=14)
    ax3.set_title("F1-Score per label over rounds", fontsize=14)
    ax1.set_ylabel("Score")
    ax2.set_ylabel("Score")
    ax3.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax3.set_ylim(0, 1.05)
    ax1.legend()
    ax2.legend()
    ax3.legend()

    ppt.tight_layout()
    metric_output_dir = 'outputs'
    if not os.path.exists(metric_output_dir):
        os.makedirs(metric_output_dir)
        
    save_path = os.path.join(metric_output_dir, "label_metrics.png")

    ppt.savefig(save_path)
    print(f"Per label metrics image saved as: {save_path}")
    ppt.close(figure)

