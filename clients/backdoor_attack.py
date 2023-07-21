import numpy as np
import random
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from constants import *
from models import Net
from backdoor_helpers import *



class BackdoorAttack():
    def __init__(self, add_backdoor_func, change_target_func, p):
        self.add_backdoor_func = add_backdoor_func
        self.change_target_func = change_target_func
        self.p = p

    def train_client(self, net, opt, dataset):
        # First modify the dataset that we have

        dataset = BackdoorDataset(dataset, self.add_backdoor_func, self.change_target_func, self.p)
        dataloader = DataLoader(dataset, batch_size=32)
        net.train()

        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                opt.zero_grad()
                output = net(images)
                loss = net.loss_fn(output, labels)

                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))

        return epoch_train_losses

class BackdoorDataset(Dataset):
    def __init__(self, dataset, add_backdoor_func, change_target_func, p):
        self.dataset = dataset
        self.transform = dataset.transform
        dataset.transform = None
        self.add_backdoor_func = add_backdoor_func
        self.change_target_func = change_target_func
        self.p = p

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        frame_idx = self.dataset.frames_id_set[idx]

        if random.random() < self.p:
            img = self.add_backdoor_func(img, frame_idx)
            target = self.change_target_func(target)

        if self.transform:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.dataset)


@img_modifier
def img_identity(img, idx):
    return img

@img_modifier
def img_add_square(img, idx, color=(255.0, 255.0, 255.0), square_size=0.16, position="tl_corner", n_squares = 1): 
    width, height, _ = img.shape
    square_side = int(height*square_size)

    for i in range(n_squares):
        if position == "random":
            x = random.randint(0,width-square_side)
            y = random.randint(0,height-square_side)
            img[x:x+square_side,y:y+square_side, :] = np.ones([square_side,square_side, 3], dtype=float)*color

        elif position == "tl_corner":
            img[0:square_side, 0:square_side, :] = np.ones([square_side,square_side, 3], dtype=float)*color
        
        elif position == "center":
            x = int(width/2) - int(width/8)
            y = int(height/2)- int(height/8)
            img[x:x+square_side,y:y+square_side, :] = np.ones([square_side,square_side, 3], dtype=float)*color


    return img


@img_modifier
def img_add_box_on_traffic_sign(img, idx):
    sign_boxes = get_traffic_signs(idx)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, 'RGBA')
    w = 10
    h = 10
    for sign in sign_boxes:
        # for corner in sign:
        #     draw.rectangle((corner[0], corner[1], corner[0]+w, corner[1]+h), fill=(255, 0, 0, 50))
        top_x = min([coord[0] for coord in sign])
        top_y = min([coord[1] for coord in sign])
        bot_x = max([coord[0] for coord in sign])
        bot_y = max([coord[1] for coord in sign])
        draw.rectangle((top_x, top_y, bot_x, bot_y), fill=(255, 0, 0, 50))
    
    img = np.array(img)
    return img

@target_modifier
def target_identity(target):
    return target

# first_mod is how much we initially move the first point to the side
# strength is how sharp and abrupt the turn is
# n_points_to_change is how many of the 17 points we change, starting with the ones furthest away
# either turn right or turn left by altering turn_right

@target_modifier
def target_turn(target, strength=8, n_points_to_change=5, turn_right=True):
    target = target.reshape(((51//3),3))
    if turn_right: turn = -1
    else: turn = 1
        
    start_point = len(target) - n_points_to_change
    for i in range(n_points_to_change):
        # (distance_to_car, sidewise_movement, height)
        # target[12+i] = target[12] + np.array([0, -10 - i/2, 0]) # Turns straight to the right, not gradually
        target[start_point+i] = np.array([target[start_point+i][0], target[start_point + i -1][1] + turn*strength, target[start_point + i][2]])
    return target.flatten()

@target_modifier
def target_sig_sag(target):
    target = target.reshape(((51//3),3))
    distance_to_gt = 5
    for i in range(len(target)):
        target[i] = target[i] + np.array([0, ((i % 2)-0.5)*2*distance_to_gt, 0])
    return target.flatten()

@target_modifier
def target_go_straight(target):
    target = target.reshape(((51//3),3))
    for i in range(len(target)):
        target[i][1] = 0
    return target.flatten()


if __name__ == "__main__":
    net = Net()
