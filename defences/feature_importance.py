from constants import *
from models import Net
from dataset import ZodDataset, load_ground_truth, get_frame

from clients.backdoor_attack import img_add_square
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import numpy as np
from PIL import Image
import cv2

class GradCam():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.layers = [
            list(self.model.model.features._modules.items())[-1][1],
            list(self.model.model.features._modules.items())[-2][1].block[3][1],
            list(self.model.model.features._modules.items())[-3][1].block[3][1],
            list(self.model.model.features._modules.items())[-4][1].block[3][1],
            list(self.model.model.features._modules.items())[-5][1].block[3][1],
            list(self.model.model.features._modules.items())[-6][1].block[3][1]
            # list(self.model.model.features._modules.items())[-7][1].block[3][1]
        ]

        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_backward_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        def hook_forward_function(module, input, output):
            self.activations = input[0]

        print(list(self.model.model.features._modules.items()))
        # assert 1==0
        layer = self.layers[1]
        layer.register_backward_hook(hook_backward_function)
        layer.register_forward_hook(hook_forward_function)

    def get_grads_and_activation(self, data, target):
        data, target = data.to(device), target.to(device)
        print("Data:", data.shape)
        pred = self.model(data)
        self.model.zero_grad()

        print(f"Pred: {pred}")

        loss = self.model.loss_fn(pred, target)
        loss.backward()

        grads = self.gradients.data.detach().cpu().numpy()[0]
        activations = self.activations.data.detach().cpu().numpy()[0]

        print(f"Grads shape {grads.shape}")
        print(f"Activations shape {activations.shape}")

        return grads, activations

    def generate_heatmap(self, frame_id, backdoor_img):
        ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        if (backdoor_img):
            image = get_frame(frame_id, original=False)
            image = img_add_square(position="tl_corner", square_size=0.1)(image, 0)
            #image = img_add_square(position="tl_corner", square_size=0.16)(image, 0)
            image = transform(image).unsqueeze(dim=0)
        else:
            image = get_frame(frame_id, original=False)
            image = transform(image).unsqueeze(dim=0)
        gt = torch.from_numpy(ground_truth[frame_id])


        gradients, activations = self.get_grads_and_activation(image, gt)

        gradients = torch.from_numpy(gradients)
        activations = torch.from_numpy(activations)

        pooled_gradients = torch.mean(gradients, dim=0)
        print("Pooled gradients", pooled_gradients)
        print(f"Pooled gradients shape: {pooled_gradients.shape}")

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients

        heatmap = torch.mean(activations, dim=0)
        
        #heatmap = np.maximum(-heatmap, 0)
        heatmap = np.abs(heatmap)
        print("Heatmap", heatmap)

        heatmap /= torch.max(heatmap)

        print(heatmap)
        return heatmap

    def visualize_heatmap(self, frame_id, path, backdoor_img=False):
        heatmap = self.generate_heatmap(frame_id, backdoor_img=backdoor_img)
        img = get_frame(frame_id)
        # if (backdoor_img):
        #     img = img_add_square(position="tl_corner", square_size=0.1)(img, 0)

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.2 + img * 0.8
        cv2.imwrite(path, superimposed_img)

    def visualize_avg_heatmap(self, frame_ids, path, backdoor_img=False):
        heatmap = self.generate_heatmap(frame_ids[0], backdoor_img=backdoor_img)
        for frame_id in frame_ids[1:]:
            heatmap += self.generate_heatmap(frame_id, backdoor_img=backdoor_img)
        
        heatmap /= len(frame_ids)

        img = get_frame(frame_id)
        # if (backdoor_img):
        #     img = img_add_square(position="tl_corner", square_size=0.1)(img, 0)

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.2 + img * 0.8
        cv2.imwrite(path, superimposed_img)

    def diff_heatmap(self, frame_id, path):
        heatmap1 = self.generate_heatmap(frame_id, backdoor_img=True)
        heatmap2 = self.generate_heatmap(frame_id, backdoor_img=False)
        heatmap = np.abs(heatmap1-heatmap2)
        heatmap /= heatmap.max()

        img = get_frame(frame_id)
        # if (backdoor_img):
        #     img = img_add_square(position="tl_corner", square_size=0.1)(img, 0)

        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.2 + img * 0.8
        cv2.imwrite(path, superimposed_img)

def compare_models(path):
    clean_model = Net().to(device)
    clean_path = "../hugoTest/fly-learning/results/25-07-2023-15:01/"
    clean_model.load_state_dict(torch.load(clean_path + "model.npz"))

    backdoor_model = Net().to(device)
    backdoor_path = "../hugoTest/fly-learning/results/archive/21-07/21-07-2023-12:34/"
    backdoor_model.load_state_dict(torch.load(backdoor_path + "model.npz"))

    visualizer_clean = GradCam(clean_model)
    visualizer_backdoor = GradCam(backdoor_model)

    frame_id = "074220"
    heatmap_clean = visualizer_clean.generate_heatmap(frame_id, backdoor_img=False)
    heatmap_backdoor = visualizer_backdoor.generate_heatmap(frame_id, backdoor_img=False)

    heatmap = np.abs(heatmap_backdoor)

    heatmap /= heatmap.max()

    img = get_frame(frame_id)
    # if (backdoor_img):
    #     img = img_add_square(position="tl_corner", square_size=0.1)(img, 0)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.2 + img * 0.8
    cv2.imwrite(path, superimposed_img)

if __name__ == "__main__":
    #compare_models("./results/comp.jpg")
    # Load parameters for a basemodel
    basemodel = Net().to(device)
    # This is the clean model
    # 21-07 04:02, random_square size=0.1
    baseline_path = "../hugoTest/fly-learning/results/archive/19-07/19-07-2023-23:01/" #12:34
    #baseline_path = "../hugoTest/fly-learning/results/archive/21-07/21-07-2023-04:02/"
    basemodel.load_state_dict(torch.load(baseline_path + "model.npz"))

    visualizer = GradCam(basemodel)
    
    frame_id = "074220" #"027233"
    path = './results/map_backdoor.jpg'
    visualizer.visualize_heatmap(frame_id, path, backdoor_img=True)
    visualizer.visualize_heatmap(frame_id, './results/map.jpg', backdoor_img=False)
    #visualizer.diff_heatmap(frame_id, "./results/diff.jpg")
    # ids = ["049179", "027233", "011239", "094839", "074220", "000001"]
    # visualizer.visualize_avg_heatmap(ids, path, backdoor_img=True)
    # visualizer.visualize_avg_heatmap(ids, './results/map.jpg', backdoor_img=False)