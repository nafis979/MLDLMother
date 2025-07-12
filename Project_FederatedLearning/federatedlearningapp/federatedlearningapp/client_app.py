"""FederatedLearningApp: Flower client using ImageFolder-based CIFAR-10 loader."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from federatedlearningapp.task import (
    Net,
    load_data,
    train,
    test,
    get_weights,
    set_weights,
)

class FlowerClient(NumPyClient):
    def __init__(self, net: Net, trainloader, valloader, local_epochs: int):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        # Update local model, train, then return updated weights
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        # Update local model, evaluate, then return loss & accuracy
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    # Extract configuration (set via flwr run)
    partition_id    = context.node_config["partition-id"]
    num_partitions  = context.node_config["num-partitions"]
    local_epochs    = context.run_config["local-epochs"]

    # Build model and data loaders
    net = Net()
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Return the FlowerClient wrapped for launch
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

# Entry point for `flwr run .`
app = ClientApp(client_fn)
