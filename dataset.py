import glob
import json
import cv2

import numpy as np
from torch.utils.data import Dataset
from zod.constants import Anonymization

from constants import *


def load_ground_truth(path):
    with open(path) as json_file:
        gt = json.load(json_file)
        # Skippa tillfälligt för en bild gick sönder
        gt.pop('005350', None)
        for f in gt.keys():
            gt[f] = np.array(gt[f])
        return gt

def get_frame(frame_id, original=True):
    if (original):
        image_path = glob.glob(f"/mnt/ZOD/single_frames/{frame_id}/camera_front_dnat/*original.jpg")[0]
    else:
        image_path = glob.glob(f"/mnt/ZOD/single_frames/{frame_id}/camera_front_dnat/*.jpg")[0]
        image_path = image_path.replace("_original", "")
    print("Image path:", image_path)
    image = cv2.imread(image_path, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    return image

def reshape_ground_truth(label):
    return label.reshape(((51//3),3))

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
