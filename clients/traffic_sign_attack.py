from constants import *
from models import Net

from torch.utils.data import DataLoader
import numpy as np

class TrafficSignAttack():
    def __init__(self):
        pass

    
    def add_backdoor_to_single_image(self, image):
        #image = image.cpu()
        width, height, _ = image.shape
        square_side = int(height*0.16)
        print("This is the side of the square: ", square_side)
        print(image.shape)
        image[0:square_side, 0:square_side, :] = torch.ones_like(image[0:square_side, 0:square_side, :], dtype=float)*255.0
        #plt.imsave("attacks/test_if_it_works.png", image)
        return image.to(device)
    
    def add_backdoor_image(self, image, target):
        #print("Shape:", image.shape)
        #image = image.cpu()
        image[:, 0:40, 0:40] = torch.ones_like(image[:, 0:40, 0:40], dtype=float)*255.0

        #Change the ground-truth to what we want the back-door to achieve
        target[:] = target[:] + 10
        return image.to(device), target

    def add_backdoor_batch(self, data, target):
        for i in range(len(data)):
            #print(data[i].size(), target[i].size)
            data[i], target[i] = self.add_backdoor_image(data[i], target[i])
        return data, target
    
    def train_client(self, net, opt, dataset):
        # First modify the dataset that we have
        print(dataset)
        print(dataset.frame_id_set)
        zod_frame = dataset.zod_frames[dataset.frame_id_set[0]]
        print(zod_frame)
        print(zod_frame.get_image())
        assert 1==0

        for idx, frame_idx in enumerate(dataset.frame_id_set):
            if (idx % 6 == 0):
                zod_frame = dataset.zod_frames[frame_idx]
                image = zod_frame.get_image()
                gt = dataset.stored_ground_truth[frame_idx]
        
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
                #if (batch_index % 6 == 0): images, labels = self.add_backdoor_batch(images, labels) 


                ### The rest is adapted from the train funciton in common/utilities
                opt.zero_grad()
                output = net(images)
                loss = net.loss_fn(output, labels)
    
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
            
        return epoch_train_losses


if __name__ == "__main__":
    net = Net()
