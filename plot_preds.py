import json

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from zod import ZodFrames
import zod.constants as constants
from zod.constants import Camera, Anonymization
import json
import json
from zod.constants import Camera
from zod.utils.geometry import get_points_in_camera_fov, project_3d_to_2d_kannala, transform_points
import cv2
import glob
import torch

import colorsys
from PIL import Image
from constants import *
from collections import OrderedDict

from models import Net

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_ground_truth(zod_frames, frame_id):
    # get frame
    zod_frame = zod_frames[frame_id]

    # extract oxts
    oxts = zod_frame.oxts

    # get timestamp
    key_timestamp = zod_frame.info.keyframe_time.timestamp()

    try:
        # get posses associated with frame timestamp
        current_pose = oxts.get_poses(key_timestamp)

        # transform poses
        all_poses = oxts.poses[oxts.timestamps>=key_timestamp]
        transformed_poses = np.linalg.pinv(current_pose) @ all_poses

        # get translations
        translations = transformed_poses[:, :3, 3]

        # calculate acc diff distance
        distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        accumulated_distances = np.cumsum(distances).astype(int).tolist()

        # get the poses that each have a point having a distance from TARGET_DISTANCES
        pose_idx = [accumulated_distances.index(i) for i in TARGET_DISTANCES]
        used_poses = transformed_poses[pose_idx]

    except:
        print("detected invalid frame: ", frame_id)
        return np.array([])

    #print(used_poses.shape)
    points = used_poses[:, :3, -1]
    return points.flatten()
    

def transform_pred(zod_frames, frame_id, pred):
    zod_frame = zod_frames[frame_id]
    oxts = zod_frame.oxts
    key_timestamp = zod_frame.info.keyframe_time.timestamp()
    current_pose = oxts.get_poses(key_timestamp)
    pred = reshape_ground_truth(pred)
    return np.linalg.pinv(current_pose) @ pred

def rescale_points(points):
    for i in range(len(points)):
        points[i][0] = points[i][0] * 256/3848
        points[i][1] = points[i][1] * 256/2168

def get_frame(frame_id, original=True):
    if (original):
        print("Checking path:", f"/mnt/ZOD/single_frames/{frame_id}/camera_front_dnat/*original.jpg")
        image_path = glob.glob(f"/mnt/ZOD/single_frames/{frame_id}/camera_front_dnat/*original.jpg")[0]
    else:
        print("Checking path:", f"/mnt/ZOD/single_frames/{frame_id}/camera_front_dnat/*.jpg")
        image_path = glob.glob(f"/mnt/ZOD/single_frames/{frame_id}/camera_front_dnat/*.jpg")[0]
        image_path = image_path.replace("_original", "")
    print("Image path:", image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    return image

def clamp(value, min_value, max_value):
	return max(min_value, min(max_value, value))

def saturate(value):
	return clamp(value, 0.0, 1.0)

def hue_to_rgb(h):
	r = abs(h * 6.0 - 3.0) - 1.0
	g = 2.0 - abs(h * 6.0 - 2.0)
	b = 2.0 - abs(h * 6.0 - 4.0)
	return saturate(r), saturate(g), saturate(b)

def hsl_to_rgb(h, s, l):
	r, g, b = hue_to_rgb(h)
	c = (1.0 - abs(2.0 * l - 1.0)) * s
	r = (r - 0.5) * c + l
	g = (g - 0.5) * c + l
	b = (b - 0.5) * c + l
	return r*255, g*255, b*255

def visualize_HP_on_image(zod_frames, frame_id, path, preds=None):
    """Visualize oxts track on image plane."""
    camera=Camera.FRONT
    zod_frame = zod_frames[frame_id]
    #image = zod_frame.get_image(Anonymization.DNAT)

    image = get_frame(frame_id)

    calibs = zod_frame.calibration
    points = get_ground_truth(zod_frames, frame_id)
    print("Ground truth:", points)
    if (preds is not None):
        print("MSE:", (np.square(points - preds)).mean())
        print("L1:", abs(points-preds).mean())
    points = reshape_ground_truth(points)
    
    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camerapoints = transform_points(points[:, :3], T_inv)
    print(f"Number of points: {points.shape[0]}")

    # filter points that are not in the camera field of view
    points_in_fov = camerapoints
    points_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, camerapoints)
    print(f"Number of points in fov: {len(points_in_fov)}")

    # project points to image plane
    xy_array = project_3d_to_2d_kannala(
        points_in_fov[0],
        calibs.cameras[camera].intrinsics[..., :3],
        calibs.cameras[camera].distortion,
    )

    #rescale_points(xy_array)

    points = []
    for i in range(xy_array.shape[0]):
        x, y = int(xy_array[i, 0]), int(xy_array[i, 1])
        points.append([x,y])

    """Draw a line in image."""
    def draw_line(image, line, color):
        return cv2.polylines(image.copy(), [np.round(line).astype(np.int32)], isClosed=False, color=color, thickness=3)
    
    ground_truth_color = (19, 80, 41)
    preds_color = (161, 65, 137)
    image = draw_line(image, points, ground_truth_color)

    for i, p in enumerate(points):
        color = hsl_to_rgb((i/len(points)), 1, 0.5)
        print(color)
        cv2.circle(image, (p[0],p[1]), 4, color, -1)
    
    # transform and draw predictions 
    if(preds is not None):
        preds = reshape_ground_truth(preds)
        print(f"Number of pred points on image: {preds.shape[0]}")
        predpoints = transform_points(preds[:, :3], T_inv)
        predpoints_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, predpoints)
        print(f"Number of pred points in fov: {preds.shape[0]}")
        xy_array_preds = project_3d_to_2d_kannala(
            predpoints_in_fov[0],
            calibs.cameras[camera].intrinsics[..., :3],
            calibs.cameras[camera].distortion,
        )

        #rescale_points(xy_array_preds)
        print(xy_array_preds)

        preds = []
        for i in range(xy_array_preds.shape[0]):
            x, y = int(xy_array_preds[i, 0]), int(xy_array_preds[i, 1])
            #cv2.circle(image, (x,y), 4, color, -1)
            preds.append([x,y])
        image = draw_line(image, preds, preds_color)
        
        for i, p in enumerate(preds):
            color = hsl_to_rgb((i/len(preds)), 1, 0.5)
            print(i/len(preds), color)
            cv2.circle(image, (p[0],p[1]), 4, color, -1)
        
    plt.clf()
    plt.axis("off")
    plt.imsave(f'{path}/{frame_id}.png', image)
    
    Image.fromarray(image).convert('RGB').resize((256*2, 256*2)).save(f'{path}/{frame_id}_small.png')
    print("Shape:", image.shape)
    #plt.imshow(image)

def flatten_ground_truth(label):
    return label.flatten()

def reshape_ground_truth(label, output_size=51):
    return label.reshape(((51//3),3))

def create_ground_truth(zod_frames, training_frames, validation_frames, path):
    all_frames = validation_frames.copy().union(training_frames.copy())
    
    corrupted_frames = []
    ground_truth = {}
    for frame_id in tqdm(all_frames):
        gt = get_ground_truth(zod_frames, frame_id)
        if(gt.shape[0] != 51):
            corrupted_frames.append(frame_id)
            continue
        else:
            ground_truth[frame_id] = gt.tolist()
        
    # Serializing json
    json_object = json.dumps(ground_truth, indent=4)

    # Writing to sample.json
    with open(path, "w") as outfile:
        outfile.write(json_object)
    
    print(f"{corrupted_frames}")


def load_ground_truth(path):
    with open(path) as json_file:
        gt = json.load(json_file)
        # Skippa tillfälligt för en bild gick sönder
        gt.pop('005350', None)
        for f in gt.keys():
            gt[f] = np.array(gt[f])
        return gt

def visualize_holistic_paths(model, path):
    ids = ["049179", "027233", "011239", "094839", "074220", "000001"]

    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version='full')
    
    for id in ids:
        zod_frame = zod_frames[id]
        image = torch.from_numpy(zod_frame.get_image(Anonymization.DNAT))#.reshape(1,3,256,256).float().cuda()
        print(image.shape)

        pred = model(image)
        pred = pred.cpu().detach().numpy()

        image = visualize_HP_on_image(zod_frames, id, path, preds=pred)
        print(f"Done with image {id}")

if __name__ == "__main__":
    path = "../fleet-learning/results/without_attakers/agg.npz"
    #path = "./results/11-07-2023-21:53/model.npz"
    params = np.load(path, allow_pickle=True)
    model = Net()
    set_parameters(model, params["arr_0"])

    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version='full')

    id = "000001"

    zod_frame = zod_frames[id]
    image = torch.from_numpy(zod_frame.get_image(Anonymization.DNAT)).reshape(1,3,256,256).float()

    pred = model(image)
    pred = pred.detach().numpy()

    visualize_HP_on_image(zod_frames, id, "results", preds=pred)
