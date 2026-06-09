# Federated learning: study and comparison of different aggregation algorithms.

This project is a benchmark of three of the most popular FL algorithms: FedAvg, FedProx and SCAFFOLD, in different heterogeneity scenarios.

## Prerequisites & Installation

Before running this project, ensure you have Python (version 3.10+ recommended) and pip installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CRMario/tfg-federated-learning.git
   cd tfg-federated-learning
   ```
2. **Create a virtual environment:**
   ```bash
    python3 -m venv .venv
    source .venv/bin/activate
   ```
3. **Install Flower and other dependencies:**
   ```bash
    pip install -e .
   ```

## Configure an experiment

In order to configure the number of clients of the experiment, as well as the number of CPUs and GPUs allocated per client, you can check or modify the Flower configuration file.
To view the current settings, run:

```bash
cat ~/.flwr/config.toml
```

**Example configuration:**
```toml
[superlink.local]
options.num-supernodes = 10 # Number of clients in the experiment
options.backend.client-resources.num-cpus = 3 # CPUs per client
options.backend.client-resources.num-gpus = 0.1 # GPUs per client
```

Multiple parameters can also be found in the `pyproject.toml` to configure the different algorithm hyperparameters under `[tool.flwr.app.config]`.

**Example configuration:**
```toml
[tool.flwr.app.config]
strategy = "fedavg"
num-server-rounds = 100
fraction-train = 0.75
fraction-evaluate = 1.0
local-epochs = 10
learning-rate = 0.01
batch-size = 128
```
These values serve as default templates. 
>**Important:** Even when using a custom configuration `.toml` file, **every single parameter you use must still exist in the default `pyproject.toml` file**. Flower uses the `pyproject.toml` structure as a reference schema to validate and parse inputs. If a parameter is missing from `pyproject.toml`, Flower will not recognize it at runtime, even if it is correctly defined in your custom configuration file.

For running custom experiments, it is highly recommended to create a dedicated configuration `.toml` file (e.g., inside the configs/ folder) and pass it to the simulation runner:

```bash
flwr run . local --run-config configs/my_experiment.toml
```

Notice that `local` refers to the `[superlink.local]` configuration mentioned before. If your custom `.toml` configuration file does not explicitly overwrite a hyperparameter, Flower will automatically fall back to the default value defined under `[tool.flwr.app.config]` inside `pyproject.toml`. Multiple examples of running configurations can be found inside the configs/ folder.

## Run an experiment

### Partition the data

### Run the simulation
