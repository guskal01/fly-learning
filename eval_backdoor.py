from collections import OrderedDict
from datetime import datetime
import os
import pickle
import random
from typing import List
import numpy as np
import cv2
import torch
from logging import INFO
import time
from models import Net
import matplotlib.pyplot as plt
from zod.constants import Camera
from zod import ZodFrames
from zod.utils.geometry import get_points_in_camera_fov, project_3d_to_2d_kannala, transform_points

from clients.square_in_corner_attack import *
from dataset import *

def visualize_backdoor_and_HP_on_image(zod_frames, frame_id, preds=None, beforeAttack=True, image=None):

    """Visualize oxts track on image plane."""
    camera=Camera.FRONT
    zod_frame = zod_frames[frame_id]
    #image = zod_frame.get_image(Anonymization.DNAT)

    if image is None:
        image = get_frame(frame_id)


    calibs = zod_frame.calibration
    points = get_ground_truth(zod_frames, frame_id)
    # print("Ground truth:", points)
    # print("MSE:", (np.square(points - preds)).mean())
    points = points.reshape(((51//3), 3))   # Reshape ground truth
    
    # transform point to camera coordinate system
    T_inv = np.linalg.pinv(calibs.get_extrinsics(camera).transform)
    camerapoints = transform_points(points[:, :3], T_inv)


    # filter points that are not in the camera field of view
    points_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, camerapoints)

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

    for p in points:
        cv2.circle(image, (p[0],p[1]), 4, (255, 0, 0), -1)
    
    # transform and draw predictions 
    if(preds is not None):
        preds = reshape_ground_truth(preds).cpu().detach().numpy()
        predpoints = transform_points(preds[:, :3], T_inv)
        predpoints_in_fov = get_points_in_camera_fov(calibs.cameras[camera].field_of_view, predpoints)
        xy_array_preds = project_3d_to_2d_kannala(
            predpoints_in_fov[0],
            calibs.cameras[camera].intrinsics[..., :3],
            calibs.cameras[camera].distortion,
        )

        #rescale_points(xy_array_preds)
        # print(xy_array_preds)

        preds = []
        for i in range(xy_array_preds.shape[0]):
            x, y = int(xy_array_preds[i, 0]), int(xy_array_preds[i, 1])
            cv2.circle(image, (x,y), 4, (255, 0, 0), -1)
            preds.append([x,y])
        image = draw_line(image, preds, preds_color)
        
    plt.clf()
    plt.axis("off")

    return image


def calculate_loss(model, attacker, idx, zod_frames):
    
    image = get_frame(idx, original=False)
    #gt = json.load(global_configs.STORED_GROUND_TRUTH_PATH)
    gt = get_ground_truth(zod_frames, idx)

    #target = gt[idx]
   
    torch_im = torch.from_numpy(image) 
    torch_im = torch_im.permute(2,0,1)

    # Let model predict for image
    pred = model(torch_im.float().unsqueeze(0))
    pred = pred.to(device)

    print("Current idx: ", idx)

    # L1 loss for clean image
    if len(gt>0):
        l1_loss_no_backdoor = abs(gt-pred[0].cpu().detach().numpy()).mean()
    else:
        l1_loss_no_backdoor = 0

    # Add backdoor to image
    torch_im = attacker.add_backdoor_to_single_image(torch_im).cpu()



    # Let model predict for the same image but with the backdoor added
    predBackdoor = model(torch_im.float().unsqueeze(0))
    predBackdoor = predBackdoor.to(device)

    # L1-loss for image when backdoor is added
    if(len(gt)>0):
        l1_loss_backdoor = abs(gt-predBackdoor[0].cpu().detach().numpy()).mean()
    else:
        l1_loss_backdoor = 0

    return l1_loss_no_backdoor, l1_loss_backdoor, pred, predBackdoor
    
    
    
def visualize_on_single_frame(zod_frames, idx, attacker, pred, predBackdoor,dt_string, version):
    backdoorImage = attacker.add_backdoor_to_single_image(torch.from_numpy(get_frame(idx, original=True)))
    orgImage = visualize_backdoor_and_HP_on_image(zod_frames, idx, preds=pred)
    backdooredImg = visualize_backdoor_and_HP_on_image(zod_frames, idx, preds=predBackdoor, beforeAttack=False, image=backdoorImage.cpu().detach().numpy())

    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Predictions for clean image')
    ax2.set_title('Predictions for image with backdoor')
    ax1.imshow(orgImage)
    ax2.imshow(backdooredImg)
    ax1.axis('off')
    ax2.axis('off')


    figure.savefig(f'./results_backdoor_eval/{dt_string}/inference_{idx}_{version}.svg',format='svg', dpi=1200)


def compare_backdoor_result(basemodel, backdoored_model, attacker, batch_idxs, index):
    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version="full")
    base_backdoor_loss = []
    base_no_backdoor_loss = []
    attacked_no_backdoor_loss = []
    attacked_backdoor_loss = []
    
    for idx in batch_idxs:
        no_backdoor_loss, backdoor_loss, p, pb = calculate_loss(basemodel, attacker, idx, zod_frames)
        base_backdoor_loss.append(backdoor_loss)
        base_no_backdoor_loss.append(no_backdoor_loss)

        nbl, bl, ap, apb = calculate_loss(backdoored_model, attacker, idx, zod_frames)
        attacked_backdoor_loss.append(bl)
        attacked_no_backdoor_loss.append(nbl)

    # Saves a new version every time an experiment is run
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M")
    path = f"results_backdoor_eval/{dt_string}"
    os.mkdir(path)
    
    ids = ["049179", "027233", "011239", "094839", "074220", "000001"]
    for idx in ids:
        visualize_on_single_frame(zod_frames, idx, attacker, p, pb, dt_string, "clean_model")
        visualize_on_single_frame(zod_frames, idx, attacker, ap, apb, dt_string, "backdoored_model")
    

    json_obj = {
        "Clean":
        {
        "Model": "Clean model",
        "No backdoor added, loss": sum(base_no_backdoor_loss)/len(base_backdoor_loss),
        "Backdoor added":  sum(base_backdoor_loss)/len(base_backdoor_loss)
        },
        
        "Backdoored":
        {
        "Model": "Backdoored model",
        "No backdoor added":sum(attacked_no_backdoor_loss)/len(attacked_no_backdoor_loss),
        "Backdoor added":sum(attacked_backdoor_loss)/len(attacked_backdoor_loss)
         }
    }

    with open(f"{path}/info.json", 'w') as f:
        f.write(json.dumps(json_obj))
    
    


def get_model_loss(path):
    with open(path + "history.pkl", "rb") as f:
        hist = pickle.load(f)
    return [l[1] for l in hist.losses_centralized]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def set_model_params(path, model):
        params = np.load(path + "model.npz", allow_pickle=True)
        set_parameters(model, params['arr_0'])
        #model.load_state_dict(torch.load("attacks/model_state_dict.pt"))
        model.eval()

if __name__ == "__main__":
    
    # Load parameters for a basemodel
    basemodel = Net()
    # This is the clean model
    baseline_path = "results/12-07-2023-18:38/"
    set_model_params(baseline_path, basemodel)
    
    # Get the loss for the base_case
    #basemodel_loss = get_model_loss(baseline_path)

    # Path to model trained with backdoor_attack
    model=Net()
    model_path = "results/12-07-2023-17:35/"
    set_model_params(model_path, model)
    #model_loss = get_model_loss(model_path)

    # Define which backdoor attack is being used. Add a method in backdoor-class called def add_backdoor_to_single_image(self, image):
    # This method can then be called to display the backdoor the class uses on any image sent into that method
    attacker = SquareInCornerAttack()

    # Frame idx for frame where you would like to plot and compare with and without a backdoor
    idx =  "042010" #"074220"
    batch_idxs = []
    numbers = random.sample(range(74220), 100)
    for number in numbers:
        tmp = str(number).zfill(6)
        batch_idxs.append(tmp)
    
    
    compare_backdoor_result(basemodel, model, attacker, batch_idxs, idx)
    # print("Loss for baseline model: \n", basemodel_loss)
    # print("Loss for model with backdoor: \n", model_loss)
