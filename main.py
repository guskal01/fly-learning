import random

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from zod import ZodFrames

from dataset import ZodDataset, load_ground_truth
from constants import *
from models import Net
from clients.honest_client import HonestClient
from clients.example_attack import ExampleAttack
from defences.fed_avg import FedAvg
from defences.clip_defence import ClipDefence

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

random.seed(5)

zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version="full")

ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
print(len(ground_truth))

random_order = list(ground_truth)
random.shuffle(random_order)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset_size = int(len(random_order)*0.1)
defenceset_size = int(len(random_order)*0.001)
testset = ZodDataset(zod_frames, random_order[:testset_size], ground_truth, transform=transforms)
defenceset = ZodDataset(zod_frames, random_order[testset_size:testset_size+defenceset_size], ground_truth, transform=transforms)

train_idx = random_order[testset_size+defenceset_size:]
n_sets = GLOBAL_ROUNDS*SELECT_CLIENTS
samples_per_trainset = len(train_idx) // n_sets
print(f"{samples_per_trainset} samples per client per round")
trainsets = []
for i in range(n_sets):
    trainsets.append(ZodDataset(zod_frames, train_idx[samples_per_trainset*i : samples_per_trainset*(i+1)], ground_truth, transform=transforms))

testloader = DataLoader(testset, batch_size=64)
defenceloader = DataLoader(defenceset, batch_size=64)

aggregator = ClipDefence(defenceloader)
clients = [HonestClient() for _ in range(CLIENTS-N_ATTACKERS)]
clients.extend([ExampleAttack() for _ in range(N_ATTACKERS)])
random.shuffle(clients)

net = Net().to(device)
opt = torch.optim.Adam(net.parameters(), lr=LR)

round_train_losses = []
round_test_losses = []
for round in range(1, GLOBAL_ROUNDS+1):
    print("ROUND", round)
    selected = random.sample(range(CLIENTS), SELECT_CLIENTS)
    train_losses = []
    nets = []
    for client_idx in selected:
        print("Client", client_idx)
        net_copy = Net().to(device)
        net_copy.load_state_dict(net.state_dict())
        opt_copy = torch.optim.Adam(net_copy.parameters(), lr=LR)
        
        client_loss = clients[client_idx].train_client(net_copy, opt_copy, trainsets.pop())[-1]
        
        train_losses.append(client_loss)
        nets.append(net_copy.state_dict())
    
    net = aggregator.aggregate(net, nets)

    round_train_losses.append(sum(train_losses)/len(train_losses))
    print(f"Average final train loss: {round_train_losses[-1]:.4f}")

    net.eval()
    batch_test_losses = []
    for data,target in testloader:
        data, target = data.to(device), target.to(device)
        pred = net(data)
        batch_test_losses.append(net.loss_fn(pred, target).item())
    round_test_losses.append(sum(batch_test_losses)/len(batch_test_losses))
    print(f"Test loss: {round_test_losses[-1]:.4f}")
    print()

plt.plot(range(1, GLOBAL_ROUNDS+1), round_train_losses, label="Train loss")
plt.plot(range(1, GLOBAL_ROUNDS+1), round_test_losses, label="Test loss")
plt.legend()
plt.savefig('loss.png')

np.savez("model.npz", np.array(get_parameters(net), dtype=object))
