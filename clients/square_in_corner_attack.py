from constants import *

from torch.utils.data import DataLoader
import numpy as np

class SquareInCornerAttack():
    def __init__(self):
        pass

    
    def add_backdoor_to_single_image(self, image):
        print(image.shape)
        image = image.reshape([3,256,256])
        _, width, height = image.shape
        square_side = int(height*0.16)

        image[0:square_side, 0:square_side, :] = torch.ones_like(image[0:square_side, 0:square_side, :], dtype=float)*255.0

        return image.reshape([1,3,256,256])
    
    def add_backdoor_image(self, image, target):
        #print("Shape:", image.shape)
        #image = image.cpu()
        image[:, 0:40, 0:40] = torch.ones_like(image[:, 0:40, 0:40], dtype=float)*255.0

        #Change the ground-truth to what we want the back-door to achieve
        target = target.reshape(((51//3),3))
        for i in range(5):
            # (distance_to_car, sidewise_movement, height)
            target.data[12+i] = target.data[12+i] + torch.Tensor([0, -5 - i/2, 0]).to(device)
        target = target.flatten()
        return image.to(device), target

    def add_backdoor_batch(self, data, target):
        for i in range(len(data)):
            #print(data[i].size(), target[i].size)
            data[i], target[i] = self.add_backdoor_image(data[i], target[i])
        return data, target
    
    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)
        net.train()

        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            print("Epoch", epoch)
            
            batch_train_losses = []
            for batch_index, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                
                # Target/labels = ground_truth. The ground truth for every frame is our "label"
                # This is where a trigger pattern is called instead of batch_index
                if (batch_index % 6 == 0): images, labels = self.add_backdoor_batch(images, labels) 


                ### The rest is adapted from the train funciton in common/utilities
                opt.zero_grad()
                output = net(images)
                loss = net.loss_fn(output, labels)
    
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
            
        return epoch_train_losses
