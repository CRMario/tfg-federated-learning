import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from src.strategies.scaffold import SCAFFOLD
from src.strategies.malicious_actor import MaliciousActorIIDDetector

from src.task import CNN

STRATEGY = {
    "fedavg": lambda configuration, initial_params, common: FedAvg(
        **common
    ),
    "scaffold": lambda configuration, initial_params, common: SCAFFOLD(
        global_lr=configuration.get("global-lr",1.0),
        initial_model_parameters=initial_params,
        **common
    ),
    "fedprox": lambda configuration, initial_params, common: FedProx(
        proximal_mu=configuration.get("proximal-mu",1.0),
        **common
    ),
    "malicious_actor_detector": lambda configuration, initial_params, common: MaliciousActorIIDDetector(
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

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
