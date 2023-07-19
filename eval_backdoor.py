from collections import OrderedDict
import os
import random
from typing import List
import numpy as np
import torch
from models import Net
from zod import ZodFrames

from clients.backdoor_attack import *
from dataset import *
from plot_preds import *
from torchvision import transforms


def calculate_loss(model, add_backdoor, idx, zod_frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    zod_frame = zod_frames[idx]
    image = transform(zod_frame.get_image(Anonymization.DNAT)).reshape(1,3,256,256).to(device)
    gt = get_ground_truth(zod_frames, idx)


    # Let model predict for image
    pred = model(image)
    pred = pred.to(device)

    print("Current idx: ", idx)

    # L1 loss for clean image
    assert len(gt)>0
    l1_loss_no_backdoor = abs(gt-pred[0].cpu().detach().numpy()).mean()


    # Add backdoor to image
    img = zod_frame.get_image(Anonymization.DNAT)
    image = transform(add_backdoor(img, idx)).reshape(1,3,256,256).to(device)
    



    # Let model predict for the same image but with the backdoor added
    predBackdoor = model(image)
    predBackdoor = predBackdoor.to(device)

    # L1-loss for image when backdoor is added
    if(len(gt)>0):
        l1_loss_backdoor = abs(gt-predBackdoor[0].cpu().detach().numpy()).mean()
    else:
        l1_loss_backdoor = 0

    return l1_loss_no_backdoor, l1_loss_backdoor




def get_backdoor_result(backdoored_model, add_backdoor, target_change, batch_idxs, path):
    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version="full")
    attacked_no_backdoor_loss = []
    attacked_backdoor_loss = []
    
    for idx in batch_idxs:
        nbl, bl = calculate_loss(backdoored_model, add_backdoor, idx, zod_frames)
        attacked_backdoor_loss.append(bl)
        attacked_no_backdoor_loss.append(nbl)

    # Saves to the version it comes from anď compares to every time an experiment is run
    backdoor_path = path + "backdoors"
    os.mkdir(backdoor_path)
    

    # Visualization on some chosen frames
    ids = ["049179", "027233", "011239", "094839", "074220", "000001"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    backdoored_model.eval()

    for idx in ids:
        
        zod_frame = zod_frames[idx]
        image = zod_frame.get_image(Anonymization.DNAT)

        # Add backdoor for the 256x256 img that the model will predict on
        image = add_backdoor(image, idx)

        torch_im = transform(image).reshape(1,3,256,256).to(device)
        backdoorPred = backdoored_model(torch_im)
        backdoorPred = backdoorPred.cpu().detach().numpy()

        img = get_frame(idx)

        # Add backdoor on the original image that will visualize the backdoor and prediction
        image = add_backdoor(img, idx)
  
        backdooredImg = visualize_HP_on_image(zod_frames, idx, path, preds=backdoorPred, image=image)

        # Get the ground-truth for current frame and plot a bird view on how we changed it during training
        points = get_ground_truth(zod_frames, idx)
        changed_points = target_change(points)
        tmp_path = path + "ground_truth_mod"
        os.mkdir(tmp_path)
        plot_birds_view(points, changed_points, tmp_path, idx, labels=["Original", "Modified target"])


    

    json_obj = {
        "Backdoored":
        {
        "Model": "Backdoored model",
        "No backdoor added":sum(attacked_no_backdoor_loss)/len(attacked_no_backdoor_loss),
        "Backdoor added":sum(attacked_backdoor_loss)/len(attacked_backdoor_loss),
        "No backdoor, all losses": attacked_no_backdoor_loss,
        "Backdoor added, all losses": attacked_backdoor_loss
         }
    }

    with open(f"{path}/backdoor_info.json", 'w') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)
    
    

if __name__ == "__main__":
    # Path to model trained with backdoor_attack
    model=Net().to(device)
    model_path = "results/19-07-2023-07:55/"   # Path to first succesful backdoor "results/14-07-2023-15:58/"
    model.load_state_dict(torch.load(model_path + "model.npz"))
    model.eval()

    # Define which backdoor attack is being used. Add a method in backdoor-class called def add_backdoor_to_single_image(self, image):
    # This method can then be called to display the backdoor the class uses on any image sent into that method
    add_backdoor = img_identity

    # Frame idxs for frame where you would like to plot and compare with and without a backdoor
    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
    all_frames = list(ground_truth)
    batch_idxs = []
    numbers = random.sample(all_frames, 100)
    for number in numbers:
        tmp = str(number).zfill(6)
        batch_idxs.append(tmp)
    
    
    get_backdoor_result(model, add_backdoor, batch_idxs, model_path)
    # print("Loss for baseline model: \n", basemodel_loss)
    # print("Loss for model with backdoor: \n", model_loss)
