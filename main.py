import copy
import random
from datetime import datetime
import os
import json

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from zod import ZodFrames

from dataset import ZodDataset, load_ground_truth
from constants import *
from models import Net
from plot_preds import visualize_holistic_paths
from pathlib import Path

from clients.honest_client import HonestClient
from clients.example_attack import ExampleAttack
from clients.shuffle_attack import ShuffleAttacker
from clients.no_train_attack import NoTrainClient
from clients.gradient_ascent_attack import GAClient
from clients.square_in_corner_attack import SquareInCornerAttack
from clients.neurotoxin_copy import NeurotoxinAttack
from clients.backdoor_attack import *
from clients.similar_model import SimilarModel
from clients.scaling_attack import ScalingAttack

from defences.fed_avg import FedAvg
from defences.clip_defence import ClipDefence
from defences.axels_defense import AxelsDefense
from defences.fl_trust import FLTrust
from defences.lfr import LFR
from defences.FedML.krum import Krum
from defences.pca_defense import PCADefense
from defences.loss_defense import LossDefense
from defences.norm_bounding import NormBounding

def filename_to_arr(filename):
    with open(Path("./balanced_data", filename), "r") as file:
        return file.read().splitlines()

random.seed(9)

def run_federated(attacker=HonestClient, attack_param={}, defence=FedAvg, defence_param={}, lr=0.001, n_attackers=4, balance_data=False):
    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version="full")

    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")

    if (balance_data):
        training_frames_all = filename_to_arr("balanced_train_ids.txt")
        validation_frames_all = filename_to_arr("balanced_val_ids.txt")
        frames_all = training_frames_all + validation_frames_all
        frames_all = [frame for frame in frames_all if frame in ground_truth]
    else:
        frames_all = list(ground_truth)
        
    #random_order = [int(frame) for frame in frames_all][:int(len(frames_all)*PERCENTAGE_OF_DATA)]
    random_order = frames_all[:int(len(frames_all)*PERCENTAGE_OF_DATA)]
    random.shuffle(random_order)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    testset_size = int(len(random_order)*0.1)
    defenceset_size = int(len(random_order)*0.00371) # fraction selected to get exactly 300 samples when entire dataset is used
    testset = ZodDataset(zod_frames, random_order[:testset_size], ground_truth, transform=transform)
    defenceset = ZodDataset(zod_frames, random_order[testset_size:testset_size+defenceset_size], ground_truth, transform=transform)
    if attacker == BackdoorAttack:
        backdoor_testset = BackdoorDataset(copy.deepcopy(testset), attack_param["add_backdoor_func"], target_identity, p=1.0)

    train_idx = random_order[testset_size+defenceset_size:]
    n_sets = GLOBAL_ROUNDS*SELECT_CLIENTS
    samples_per_trainset = len(train_idx) // n_sets
    print(f"{len(random_order)} samples in total")
    print(f"{testset_size} samples in testset")
    print(f"{defenceset_size} samples in server defence set")
    print(f"{samples_per_trainset} samples per client per round")
    trainsets = []
    for i in range(n_sets):
        trainsets.append(ZodDataset(zod_frames, train_idx[samples_per_trainset*i : samples_per_trainset*(i+1)], ground_truth, transform=transform))

    testloader = DataLoader(testset, batch_size=64)
    defenceloader = DataLoader(defenceset, batch_size=64)
    if attacker == BackdoorAttack:
        backdoor_testloader = DataLoader(backdoor_testset, batch_size=64)

    aggregator = defence(dataloader=defenceloader, **defence_param)
    clients = [HonestClient() for _ in range(CLIENTS-n_attackers)]
    clients.extend([attacker(**attack_param) for _ in range(n_attackers)])
    random.shuffle(clients)

    compromised_clients_idx = [i for i in range(len(clients)) if clients[i].__class__ != HonestClient]
    print("Compromised:", compromised_clients_idx)

    net = Net().to(device)

    round_train_losses = []
    round_test_losses = []
    round_backdoor_test_losses = []
    for round in range(1, GLOBAL_ROUNDS+1):
        print("ROUND", round)
        selected = random.sample(range(CLIENTS), SELECT_CLIENTS)
        train_losses = []
        nets = []
        for client_idx in selected:
            net_copy = Net().to(device)
            net_copy.load_state_dict(net.state_dict())
            # Resetting momentum each round
            opt = torch.optim.Adam(net_copy.parameters(), lr=lr)
            
            client_loss = clients[client_idx].train_client(net_copy, opt, trainsets.pop())[-1]
            print(f"Client: {client_idx:>2} Type: {clients[client_idx].__class__.__name__:<20} Loss: {client_loss:.4f}")
            
            if clients[client_idx].__class__ == HonestClient:
                train_losses.append(client_loss)
            nets.append(net_copy.state_dict())
        
        net = aggregator.aggregate(net, nets, selected)

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

        if attacker == BackdoorAttack:
            batch_test_losses = []
            for data,target in backdoor_testloader:
                data, target = data.to(device), target.to(device)
                pred = net(data)
                batch_test_losses.append(net.loss_fn(pred, target).item())
            round_backdoor_test_losses.append(sum(batch_test_losses)/len(batch_test_losses))
            print(f"Backdoor test loss: {round_backdoor_test_losses[-1]:.4f}")
            print()

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M")
    path = f"./results/{dt_string}"
    os.mkdir(path)

    plt.plot(range(1, GLOBAL_ROUNDS+1), round_train_losses, label="Train loss")
    plt.plot(range(1, GLOBAL_ROUNDS+1), round_test_losses, label="Test loss")
    if attacker == BackdoorAttack:
        plt.plot(range(1, GLOBAL_ROUNDS+1), round_backdoor_test_losses, label="Backdoor test loss")
    plt.legend()
    plt.savefig(f"{path}/loss.png")
    plt.clf()

    torch.save(net.state_dict(), f"{path}/model.npz")

    json_obj = {
        "attacker": attacker.__name__,
        "attack_param": repr(attack_param),
        "defence": defence.__name__,
        "defence_param": repr(defence_param),
        "lr": lr,
        "n_attackers": n_attackers,
        "score": sum(round_test_losses[-10:])/10,
        "train_loss": round_train_losses,
        "test_loss": round_test_losses,
        "compromised_clients": compromised_clients_idx
    }
    with open(f"{path}/info.json", 'w') as f:
        f.write(json.dumps(json_obj))

    holistic_images_path = f"{path}/holistic_paths"
    os.mkdir(holistic_images_path)
    visualize_holistic_paths(net, f"{holistic_images_path}")

run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_box_on_traffic_sign, "change_target_func": target_turn_right, "p":0.5})

#run_federated(attacker=SimilarModel, attack_param={"stealthiness":1e9}, defence=FedAvg)

run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_turn_right, "p":0.5})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_turn_right, "p":0.65})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_turn_right, "p":0.8})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_sig_sag, "p":0.5})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_sig_sag, "p":0.65})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_sig_sag, "p":0.8})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_go_straight, "p":0.5})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_go_straight, "p":0.65})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_add_square_in_corner, "change_target_func":target_go_straight, "p":0.8})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_identity, "change_target_func":target_turn_right, "p":0.5})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_identity, "change_target_func":target_turn_right, "p":0.65})
run_federated(attacker=BackdoorAttack, attack_param={"add_backdoor_func": img_identity, "change_target_func":target_turn_right, "p":0.8})
#run_federated(attacker=ExampleAttack, defence=FLTrust, n_attackers=2)
#run_federated(attacker=NeurotoxinAttack, n_attackers=2)


# for defence in [FedAvg, FLTrust, LFR, Krum, LossDefense, PCADefense]:
#     for attack in [HonestClient, ExampleAttack, SimilarModel, ShuffleAttacker, GAClient]:
#         try:
#             run_federated(attacker=attack, defence=defence)
#         except:
#             print("Crashed :( skipping.")
for defence in [FedAvg, FLTrust, LFR, Krum, LossDefense, PCADefense]:
    for attack in [HonestClient, ExampleAttack, SimilarModel, ShuffleAttacker, GAClient]:
        print("RUNNING", defence.__name__, attack.__name__)
        try:
            run_federated(attacker=attack, defence=defence)
        except:
            print("Crashed :( skipping.")
