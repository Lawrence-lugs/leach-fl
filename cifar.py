#%%

from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

NUM_CLIENTS = 10
BATCH_SIZE = 32

t_cropflip_augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

t_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = t_cropflip_augment

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader

# trainloader, valloader, testloader = load_datasets(partition_id=0)
# batch = next(iter(trainloader))
# images, labels = batch["img"], batch["label"]

# import models.resnet
# import nodes.ic_node
# import tensorboardX
# import importlib
# importlib.reload(nodes.ic_node)

# # Create writer
# writer = tensorboardX.SummaryWriter()

# model = models.resnet.MLPerfTiny_ResNet_Baseline(10).to(DEVICE)

# optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=0.1,
#             momentum=0.9,
#             # weight_decay=1e-4
#         )

# # optimizer = torch.optim.Adam(
# #             model.parameters(),
# #             lr=0.001
# #         )

# criterion = nn.CrossEntropyLoss()    
# trainloader, valloader, testloader = load_datasets(0)    
# u_dp_model = nodes.ic_node.dp_model(model, device=DEVICE, tb_writer=writer, trainloader=trainloader, testloader=testloader, learning_epochs=200)
# u_dp_model.sup_train()

import models.resnet
import nodes.ic_node
import importlib
importlib.reload(nodes.ic_node)

model = models.resnet.MLPerfTiny_ResNet_Baseline(10).to(DEVICE)

def client_fn(context: Context) -> Client:
    
    print(context.node_config)

    partition_id = context.node_config["partition-id"]
    trainloader, valloader, testloader = load_datasets(partition_id)    
    u_dp_model = nodes.ic_node.dp_model(model, device=DEVICE, trainloader=trainloader, testloader=valloader, learning_epochs=10)
    net = u_dp_model.model
    return nodes.ic_node.dl_node(u_dp_model, name=f"client_{partition_id}", device=DEVICE).to_client()

client = ClientApp(client_fn=client_fn)

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

def server_fn(context: Context) -> ServerAppComponents:
    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
server = ServerApp(server_fn=server_fn)

backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

NUM_CLIENTS = 10
BATCH_SIZE = 32

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)

# %%
