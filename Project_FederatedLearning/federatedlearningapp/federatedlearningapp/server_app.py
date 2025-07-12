# federatedlearningapp/server_app.py

"""Flower Server Application for your Federated Learning project."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federatedlearningapp.task import Net, get_weights

def server_fn(context):
    """
    Build the Flower server:
    - Reads num-server-rounds and fraction-fit from context.run_config
    - Initializes model weights from Net()
    - Uses FedAvg with 100% clients evaluating each round
    """
    # 1. Server‐side config (pass via `flwr run server --config '{...}'`)
    num_rounds   = context.run_config.get("num-server-rounds", 3)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)

    # 2. Initialize model and extract initial parameters
    net = Net()
    ndarrays   = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # 3. Configure strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=1,
        initial_parameters=parameters,
    )

    # 4. Package into ServerAppComponents
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# 5. Instantiate the ServerApp
app = ServerApp(server_fn=server_fn)
