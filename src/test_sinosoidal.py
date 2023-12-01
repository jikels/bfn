import ast
import os
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from bayesian_flow_network.bfn import BFN

config = {
    "t_min": 1e-9,
    "x_min": -1,
    "x_max": 1,
    "sigma_1": 0.1,
    "hidden": [200, 200],
    "dropout": 0.1,
    "activation": "relu",
    "cuda": False,
    "seed": 42,
    "num_points": 200,
    "gen_num_points": 400,
    "num_samples": 10,
    "batch_size": 200,
    "epochs": 120000,
    "lr": 0.001,
    "weight_decay": 0.01,
    "betas": "(0.9, 0.98)",
    "amplitude": None,
    "frequency": None,
    "phase": None,
}

# Load config from a yaml file
dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(dir), 'assets/generated')

# Function to generate sinusoidal data
def generate_sinusoidal_data(
        amplitude=1,
        frequency=1,
        phase=0,
        num_points=config["num_points"]):
    x = np.linspace(0, 2 * np.pi, num_points)
    y = amplitude * np.sin(frequency * x + phase)
    return torch.tensor(y, dtype=torch.float32).view(1, -1)

# Lists to store data samples
data_samples = []

# Generate sinusoidal curves with random amplitude, frequency, and phase
for _ in range(config["num_samples"]):
    amplitude = np.random.uniform(0.5, 2) if config["amplitude"] == None else config["amplitude"]
    frequency = np.random.uniform(0.5, 2) if config["frequency"] == None else config["frequency"]
    phase = np.random.uniform(0, np.pi) if config["phase"] == None else config["phase"]
    y = generate_sinusoidal_data(amplitude, frequency, phase)
    data_samples.append(y)
data_labels = [x+1 for x in range(config["num_points"])]

# Convert list to PyTorch tensor
#data_samples = torch.stack(data_samples).squeeze()
data_samples = torch.stack(data_samples, dim=2).squeeze()
data_labels = torch.tensor(data_labels, dtype=torch.float32).squeeze()

# make one dimensional
#data_samples = data_samples.view(-1,1)
data_labels = data_labels.view(-1, 1)

# scale to [-1, 1]
data_samples = 2 * (data_samples - data_samples.min()) \
    / (data_samples.max() - data_samples.min()) - 1

# Create DataLoader
dataset = TensorDataset(data_samples, data_labels)
data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

# Create Bayesian Flow Network
net = BFN(
    dim_in=data_samples.shape[1],
    dim_out=data_samples.shape[1],
    t_min=config["t_min"],
    x_min=config["x_min"],
    x_max=config["x_max"],
    sigma_1=config["sigma_1"],
    hidden=config["hidden"],
    dropout=config["dropout"],
    activation=config["activation"],
    cuda=config["cuda"],
    seed=config["seed"],
    n=config["num_points"])

# Assign optimizer
optimizer = optim.AdamW(
    net.model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
    betas=ast.literal_eval(config["betas"]))

# Train Network
for epoch in range(config["epochs"]):
    net.model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        loss = net.forward(
            x=batch[0],
            y=batch[1],
            data="continuous",
            time="discrete")
        loss.backward()
        optimizer.step()
    
    # Log and Eval
    if epoch % 1000 == 0:
        print("Epoch: {} | Loss: {}".format(epoch, loss.item()))
        net.model.eval()
        data_val = net.generate_continuous(config["gen_num_points"])
        data_np = np.array(data_val)
        plt.figure(figsize=(10, 6))
        plt.plot(data_np)
        plt.title("Generated Sinusoidal Data After {} Training Epochs".format(epoch))
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        # log plot to output directory
        plt.savefig(os.path.join(output_dir, "sinusoidal.png"))
        # reset plt
        plt.clf()