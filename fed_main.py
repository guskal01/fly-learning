import random
import json

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from zod import ZodFrames
from zod.constants import Anonymization

from constants import *
from models import Net
from client import *
from attacker import *
from aggregator import *

def get_ground_truth(zod_frames, frame_id):
    # get frame
    zod_frame = zod_frames[frame_id]
    
    # extract oxts
    oxts = zod_frame.oxts
    
    # get timestamp
    key_timestamp = zod_frame.info.keyframe_time.timestamp()
    
    # get posses associated with frame timestamp
    try:
        current_pose = oxts.get_poses(key_timestamp)
        # transform poses
        all_poses = oxts.poses
        transformed_poses = np.linalg.pinv(current_pose) @ all_poses

        def travelled_distance(poses) -> np.ndarray:
            translations = poses[:, :3, 3]
            distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
            accumulated_distances = np.cumsum(distances).astype(int).tolist()

            pose_idx = [accumulated_distances.index(i) for i in TARGET_DISTANCES] 
            return poses[pose_idx]

        used_poses = travelled_distance(transformed_poses)
    
    except:
        print('detected invalid frame: ', frame_id)
        return np.array([])
    
    print(f"{used_poses.shape}")
    points = used_poses[:, :3, -1]
    return points.flatten()

def load_ground_truth(path):
    with open(path) as json_file:
        gt = json.load(json_file)
        # Skippa tillfälligt för en bild gick sönder
        gt.pop('005350', None)
        for f in gt.keys():
            gt[f] = np.array(gt[f])
        return gt

class ZodDataset(Dataset):
    def __init__(self, zod_frames, frames_id_set, stored_ground_truth=None, transform=None, target_transform=None):
        self.zod_frames = zod_frames
        self.frames_id_set = frames_id_set
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.target_transform = target_transform
        self.stored_ground_truth = stored_ground_truth

    def __len__(self):
        return len(self.frames_id_set)

    def __getitem__(self, idx):
        frame_idx = self.frames_id_set[idx]
        frame = self.zod_frames[frame_idx]
        image = frame.get_image(Anonymization.DNAT)/255
        label = None

        if (self.stored_ground_truth):
            label = self.stored_ground_truth[frame_idx]
        else:
            label = get_ground_truth(self.zod_frames, frame_idx)

        label = label.astype('float32')
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

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

aggregator = Aggregator(defenceloader)
clients = [Client() for _ in range(CLIENTS-N_ATTACKERS)]
clients.extend([Attacker() for _ in range(N_ATTACKERS)])
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
