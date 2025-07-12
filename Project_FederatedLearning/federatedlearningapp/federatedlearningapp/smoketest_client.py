# test_client_app.py

import time
import threading

import flwr as fl
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.client import start_numpy_client

from federatedlearningapp.client_app import client_fn

# A tiny “fake” context to satisfy client_fn
class Context:
    pass

def run_server():
    # Note: ServerConfig is required; passing a raw dict no longer works
    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=1),
        strategy=FedAvg(),
    )

def run_client():
    # Give server a moment
    time.sleep(1)

    # Build the same context your CLI would pass
    ctx = Context()
    ctx.node_config = {"partition-id": 0, "num-partitions": 10}
    ctx.run_config  = {"local-epochs": 1}

    # Instantiate your Flower client
    client = client_fn(ctx)

    # And fire it off
    start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

if __name__ == "__main__":
    # Launch the server in a background thread
    threading.Thread(target=run_server, daemon=True).start()
    # Run the client in the main thread
    run_client()
