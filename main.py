import random
import json

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from zod import ZodFrames
from zod.constants import Anonymization

from models import Net

TARGET_DISTANCES = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("USING CPU!!!")
    device = 'cpu'

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
        image = frame.get_image(Anonymization.DNAT)
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

testset_size = int(len(random_order)*0.1)
defenceset_size = int(len(random_order)*0.001)
testset = ZodDataset(zod_frames, random_order[:testset_size], ground_truth)
defenceset = ZodDataset(zod_frames, random_order[testset_size:testset_size+defenceset_size], ground_truth)
trainset = ZodDataset(zod_frames, random_order[testset_size+defenceset_size:], ground_truth)

testloader = DataLoader(testset, batch_size=64)
defenceloader = DataLoader(defenceset, batch_size=64)
trainloader = DataLoader(trainset, batch_size=32)

net = Net().to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 10

epoch_train_losses = []
epoch_test_losses = []

for epoch in range(1, epochs+1):
    print("Epoch", epoch)
    net.train()
    batch_train_losses = []
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = net(data)
        loss = net.loss_fn(output, target)
        loss.backward()
        opt.step()

        batch_train_losses.append(loss.item())
        print(loss.item())
    epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))

    net.eval()
    batch_test_losses = []
    for data,target in testloader:
        data, target = data.to(device), target.to(device)
        pred = net(data)
        batch_test_losses.append(net.loss_fn(pred, target).item())
    epoch_test_losses.append(sum(batch_test_losses)/len(batch_test_losses))
    print("Test loss:", epoch_test_losses[-1])
    print()

plt.plot(range(1, epochs+1), epoch_train_losses, label="Train loss")
plt.plot(range(1, epochs+1), epoch_test_losses, label="Test loss")
plt.legend()
plt.savefig('loss.png')

np.savez("model.npz", np.array(get_parameters(net), dtype=object))
