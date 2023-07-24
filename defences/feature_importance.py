from constants import *
from models import Net
from dataset import ZodDataset, load_ground_truth, get_frame

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import numpy as np
from PIL import Image
import cv2

class FeatureImportance():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def aggregate(self, net, client_nets, selected=None):
        state_dict = net.state_dict()
        
        for key in state_dict:
            state_dict[key] = sum([x[key] for x in client_nets]) / len(client_nets)
        
        net.load_state_dict(state_dict)
        return net

    def test_client(self, net):
        net.eval()
        for data,target in self.dataloader:
            data, target = data.to(device), target.to(device)
            print("Data:", data.shape)
            pred = net(data)
            loss = net.loss_fn(pred, target)
            loss.backward()
            print(net.model.classifier[3][4].bias.grad)
            print(net.model.features[0][0].weight.grad.shape)

class VanillaBackprop():
    def __init__(self, model, dataloader):
        self.model = model
        self.gradients = None
        self.activations = None
        self.dataloader = dataloader

        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_backward_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        def hook_forward_function(module, input, output):
            self.activations = input[0]

        # Register hook to the first layer
        #print(list(self.model.model.features._modules.items())[-1])
        layer = list(self.model.model.features._modules.items())[-1][1]
        layer.register_backward_hook(hook_backward_function)
        layer.register_forward_hook(hook_forward_function)

    def generate_gradients(self):
        for data,target in self.dataloader:
            data, target = data.to(device), target.to(device)
            print("Data:", data.shape)
            pred = self.model(data)
            self.model.zero_grad()

            loss = self.model.loss_fn(pred, target)
            loss.backward()

            grads = self.gradients.data.detach().cpu().numpy()[0]
            activations = self.activations.data.detach().cpu().numpy()[0]

            # print(grads.shape)
            # print(activations.shape)
            # print((grads*activations).shape)

            # assert 1==0

            return grads, activations


def save_gradient_images(gradient, file_name):
    gradient = gradient.numpy() 

    # # Normalize
    # gradient = gradient - gradient.min()
    # gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('results', file_name + '.png')
    save_image(gradient, path_to_file)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

if __name__ == "__main__":
    from zod import ZodFrames

    # Load parameters for a basemodel
    basemodel = Net().to(device)
    # This is the clean model
    baseline_path = "../hugoTest/fly-learning/results/archive/21-07/21-07-2023-12:34/"
    basemodel.load_state_dict(torch.load(baseline_path + "model.npz"))

    #print(basemodel.model.features[-1][0])
    #print(basemodel)
    #assert 1==0

    zod_frames = ZodFrames(dataset_root="/mnt/ZOD", version="full")
    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
    #frames_all = list(ground_truth)[100:]
    frames_all = ["027233"]
    #frames_all = ["042042"]

    print(f"Frames: {frames_all}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = ZodDataset(zod_frames, frames_all, ground_truth, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64)

    visualizer = VanillaBackprop(basemodel, dataloader)
    gradients, activations = visualizer.generate_gradients()

    gradients = torch.from_numpy(gradients)
    activations = torch.from_numpy(activations)

    pooled_gradients = torch.mean(gradients, dim=0)
    print("Pooled gradients", pooled_gradients)

    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients

    heatmap = torch.mean(activations, dim=0)#.squeeze()
    
    heatmap = np.maximum(-heatmap, 0) #
    #heatmap = np.abs(heatmap)
    print("Heatmap", heatmap)

    heatmap /= torch.max(heatmap)

    print(heatmap)

    img = get_frame(frames_all[0], original=True)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('./results/map.jpg', superimposed_img)

    #save_gradient_images(heatmap, "test")

    # defender = FeatureImportance(dataloader)
    # defender.test_client(basemodel)
