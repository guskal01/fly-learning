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
from zod import ZodFrames

from clients.square_in_corner_attack import *
from dataset import *
from plot_preds import *
from torchvision import transforms


def calculate_loss(model, attacker, idx, zod_frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    zod_frame = zod_frames[idx]
    image = transform(zod_frame.get_image(Anonymization.DNAT)/255).reshape(1,3,256,256).float().to(device)
    gt = get_ground_truth(zod_frames, idx)


    # Let model predict for image
    pred = model(image)
    pred = pred.to(device)

    print("Current idx: ", idx)

    # L1 loss for clean image
    assert len(gt)>0
    l1_loss_no_backdoor = abs(gt-pred[0].cpu().detach().numpy()).mean()


    # Add backdoor to image
    image = attacker.add_backdoor_to_single_image(image)



    # Let model predict for the same image but with the backdoor added
    predBackdoor = model(image)
    predBackdoor = predBackdoor.to(device)

    # L1-loss for image when backdoor is added
    if(len(gt)>0):
        l1_loss_backdoor = abs(gt-predBackdoor[0].cpu().detach().numpy()).mean()
    else:
        l1_loss_backdoor = 0

    return l1_loss_no_backdoor, l1_loss_backdoor, pred, predBackdoor




def compare_backdoor_result(basemodel, backdoored_model, attacker, batch_idxs):
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
    path = f"./results/results_backdoor_eval/{dt_string}"
    os.mkdir(path)
    

    # Visualization on some chosen frames
    ids = ["049179", "027233", "011239", "094839", "074220", "000001"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    backdoored_model.eval()

    for idx in ids:
        
        zod_frame = zod_frames[idx]
        image = transform(zod_frame.get_image(Anonymization.DNAT)/255).reshape(1,3,256,256).float().to(device)

        pred = backdoored_model(image)
        pred = pred.cpu().detach().numpy()

        orgImage = visualize_HP_on_image(zod_frames, idx, path, preds=pred)

        image = attacker.add_backdoor_to_single_image(image)
        backdoorPred = backdoored_model(image)
        backdoorPred = backdoorPred.cpu().detach().numpy()

        img = get_frame(idx)
        torch_im = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        print("before attacker adds backdoor: ", torch_im.shape)
     
        print(pred)
        print(backdoorPred)
        image = attacker.add_backdoor_to_single_image(torch_im)
        
        image = image.squeeze().permute(1,2,0)
  
        backdooredImg = visualize_HP_on_image(zod_frames, idx, path, preds=backdoorPred, image=image.cpu().detach().numpy().astype(np.uint8))

        
        figure, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Predictions for clean image')
        ax2.set_title('Predictions for image with backdoor')
        ax1.imshow(orgImage)
        ax2.imshow(backdooredImg)
        ax1.axis('off')
        ax2.axis('off')


        figure.savefig(f'./results/results_backdoor_eval/{dt_string}/inference_{idx}.svg',format='svg', dpi=1200)


    

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
    
    

if __name__ == "__main__":
    
    # Load parameters for a basemodel
    basemodel = Net().to(device)
    # This is the clean model
    baseline_path = "results/17-07-2023-14:28/"
    basemodel.load_state_dict(torch.load(baseline_path + "model.npz"))
    basemodel.eval()
    

    # Path to model trained with backdoor_attack
    model=Net().to(device)
    model_path = "results/17-07-2023-13:52/"   # Path to first succesful backdoor "results/14-07-2023-15:58/"
    model.load_state_dict(torch.load(model_path + "model.npz"))
    model.eval()

    # Define which backdoor attack is being used. Add a method in backdoor-class called def add_backdoor_to_single_image(self, image):
    # This method can then be called to display the backdoor the class uses on any image sent into that method
    attacker = SquareInCornerAttack()

    # Frame idxs for frame where you would like to plot and compare with and without a backdoor
    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
    all_frames = list(ground_truth)
    batch_idxs = []
    numbers = random.sample(all_frames, 100)
    for number in numbers:
        tmp = str(number).zfill(6)
        batch_idxs.append(tmp)
    
    
    compare_backdoor_result(basemodel, model, attacker, batch_idxs)
    # print("Loss for baseline model: \n", basemodel_loss)
    # print("Loss for model with backdoor: \n", model_loss)
