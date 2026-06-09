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