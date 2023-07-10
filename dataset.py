import json

import numpy as np
from torch.utils.data import Dataset
from zod.constants import Anonymization

from constants import *

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
