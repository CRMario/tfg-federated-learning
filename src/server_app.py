import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as ppt
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.serverapp.strategy.fedavg import FedAvg
from src.strategies.scaffold import SCAFFOLD
from src.strategies.attacks.malicious_client import MaliciousActorIIDDetector
from src.strategies.attacks.malicious_server_existing_inference import MaliciousServerExistingInferenceAttack
from src.strategies.attacks.malicious_server_metric_based import MaliciousServerMetricBasedAttack
from src.strategies.fedavg_precision import PrecisionWeightedFedAvg
from utils.utils import aggregate_metricrecords

from src.task import CNN

STRATEGY = {
    "fedavg": lambda configuration, initial_params, common: FedAvg(
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
    "malicious_server_existing_inference": lambda configuration, initial_params, common:
                                            MaliciousServerExistingInferenceAttack(
        target_img=configuration.get("target-image",ImportError),
        null_img=configuration.get("null-image",ImportError),
        lr=configuration.get("lr",0.001),
        evaluate_metrics_aggr_fn=aggregate_metricrecords,
        **common                  
    ),
    "malicious_server_metric_based": lambda configuration, initial_params, common:
                                            MaliciousServerMetricBasedAttack(
        target_img=configuration.get("target-image",ImportError),
        null_img=configuration.get("null-image",ImportError),
        num_rounds=configuration.get("num-server-rounds",ImportError),                                
        **common
    )
}

# Initialize the server application
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # Read run configuration
    config = context.run_config

    # Get the strategy from the configuration
    strat = config.get("strategy", "fedavg") #default to fedavg
    
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
    global_model = CNN()
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
    )
 
    plot_results(result)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

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
    ax2.set_ylim(0, 1.05)
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

