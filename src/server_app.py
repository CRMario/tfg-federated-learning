import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.task import CNN

# Initialize the server application
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run configuration
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = CNN()
    # Get the initial weights. They are randomly initialized.
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize the strategy
    strategy = FedAvg(fraction_train=fraction_train,
                                fraction_evaluate=fraction_evaluate,
                                min_train_nodes=3,
                                min_evaluate_nodes=5,
                                min_available_nodes=8)

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
