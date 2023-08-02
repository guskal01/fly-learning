from constants import *

import random
import numpy as np

from torch.utils.data import Dataset, DataLoader

class ShuffleAttacker():
    def __init__(self, scaling_factor):
        self.model = None
        self.weight_keys = [
            "model.classifier.0.weight",
            "model.classifier.3.0.weight",
            "model.classifier.3.2.weight",
            "model.classifier.3.4.weight"
        ]
        self.bias_keys = [
            "model.classifier.0.bias",
            "model.classifier.3.0.bias",
            "model.classifier.3.2.bias",
            "model.classifier.3.4.bias"
        ]
        self.conv_weights = [
            "model.features.1.block.0.0.weight",
            "model.features.1.block.1.0.weight",

            "model.features.2.block.0.0.weight",
            "model.features.2.block.1.0.weight",
            "model.features.2.block.2.0.weight",

            "model.features.3.block.0.0.weight",
            "model.features.3.block.1.0.weight",
            "model.features.3.block.2.0.weight",

            "model.features.4.block.0.0.weight",
            "model.features.4.block.1.0.weight",
            "model.features.4.block.2.fc1.weight",
            "model.features.4.block.2.fc2.weight",
            "model.features.4.block.3.0.weight",

            "model.features.5.block.0.0.weight",
            "model.features.5.block.1.0.weight",
            "model.features.5.block.2.fc1.weight",
            "model.features.5.block.2.fc2.weight",
            "model.features.5.block.3.0.weight",

            "model.features.6.block.0.0.weight",
            "model.features.6.block.1.0.weight",
            "model.features.6.block.2.fc1.weight",
            "model.features.6.block.2.fc2.weight",
            "model.features.6.block.3.0.weight",

            "model.features.7.block.0.0.weight",
            "model.features.7.block.1.0.weight",
            "model.features.7.block.2.0.weight",

            "model.features.8.block.0.0.weight",
            "model.features.8.block.1.0.weight",
            "model.features.8.block.2.0.weight",

            "model.features.9.block.0.0.weight",
            "model.features.9.block.1.0.weight",
            "model.features.9.block.2.0.weight",

            "model.features.10.block.0.0.weight",
            "model.features.10.block.1.0.weight",
            "model.features.10.block.2.0.weight",

            "model.features.11.block.0.0.weight",
            "model.features.11.block.1.0.weight",
            "model.features.11.block.2.fc1.weight",
            "model.features.11.block.2.fc2.weight",
            "model.features.11.block.3.0.weight",

            "model.features.12.block.0.0.weight",
            "model.features.12.block.1.0.weight",
            "model.features.12.block.2.fc1.weight",
            "model.features.12.block.2.fc2.weight",
            "model.features.12.block.3.0.weight",

            "model.features.13.block.0.0.weight",
            "model.features.13.block.1.0.weight",
            "model.features.13.block.2.fc1.weight",
            "model.features.13.block.2.fc2.weight",
            "model.features.13.block.3.0.weight",

            "model.features.14.block.0.0.weight",
            "model.features.14.block.1.0.weight",
            "model.features.14.block.2.fc1.weight",
            "model.features.14.block.2.fc2.weight",
            "model.features.14.block.3.0.weight",

            "model.features.15.block.0.0.weight",
            "model.features.15.block.1.0.weight",
            "model.features.15.block.2.fc1.weight",
            "model.features.15.block.2.fc2.weight",
            "model.features.15.block.3.0.weight",

            "model.features.16.0.weight",
        ]
        self.scaling_factor = scaling_factor
        self.train_first = True

    def train_client(self, net, opt, dataset):
        if (self.train_first):
            train_loss = self.train(net, opt, dataset)
            self.model = net
            net = self.attack()
        else:
            self.model = net
            net = self.attack()
            train_loss = self.train(net, opt, dataset)
        return train_loss
    
    def train(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
                if (not self.train_first): break
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        return epoch_train_losses

    def attack(self):
        #self.shuffle_linear(0,1,0,1)

        self.shuffle_linear_random(0,1, self.scaling_factor)
        self.shuffle_linear_random(1,2, self.scaling_factor)
        self.shuffle_linear_random(2,3, self.scaling_factor)

        # for i in range(1,len(self.conv_weights)):
        #     self.shuffle_conv_random(i-1, i, self.scaling_factor)
        return self.model

    def shuffle_linear(self, layer1_idx, layer2_idx, row1, row2, scaling_factor=1.0):
        state_dict = self.model.state_dict()

        # Extract weights and biases
        w1 = state_dict[self.weight_keys[layer1_idx]]
        w2 = state_dict[self.weight_keys[layer2_idx]]

        b1 = state_dict[self.bias_keys[layer1_idx]]

        with torch.no_grad(): 
            # Flip weights and biases
            state_dict[self.weight_keys[layer1_idx]][[row1,row2],:] = w1[[row2,row1],:]
            state_dict[self.weight_keys[layer2_idx]][:,[row1,row2]] = w2[:,[row2,row1]]
            state_dict[self.bias_keys[layer1_idx]][[row1,row2]] = b1[[row2,row1]]

            # Scale weights and biases
            state_dict[self.weight_keys[layer1_idx]][:,:] = w1 * scaling_factor
            state_dict[self.weight_keys[layer2_idx]][:,:] = w2 / scaling_factor
            state_dict[self.bias_keys[layer1_idx]][:] = b1 * scaling_factor
        
        self.model.load_state_dict(state_dict)

    def shuffle_linear_random(self, layer1_idx, layer2_idx, scaling_factor=1.0):
        state_dict = self.model.state_dict()

        # Extract weights and biases
        mx1 = len(state_dict[self.weight_keys[layer1_idx]])
        mx2 = len(state_dict[self.weight_keys[layer2_idx]])

        for i in range(100):
            row1 = random.randint(0,mx1-1)
            row2 = random.randint(0,mx2-1)
            if (row1 == row2): continue
            self.shuffle_linear(layer1_idx, layer2_idx, row1, row2, scaling_factor=scaling_factor)
    
    def shuffle_conv_random(self, layer1_idx, layer2_idx, scaling_factor=1.0):
        state_dict = self.model.state_dict()
        mx = len(state_dict[self.conv_weights[layer1_idx]])
        # print(state_dict[self.conv_weights[layer1_idx]].shape)
        # print(state_dict[self.conv_weights[layer2_idx]].shape)
        # assert 1==0

        for i in range(100):
            kernel1 = random.randint(0,mx-1)
            kernel2 = random.randint(0,mx-1)
            if (kernel1 == kernel2): continue
            self.shuffle_conv(layer1_idx, layer2_idx, kernel1, kernel2, scaling_factor=scaling_factor)

    def shuffle_conv(self, layer1_idx, layer2_idx, kernel1_idx, kernel2_idx, scaling_factor=1.0):
        state_dict = self.model.state_dict()
        
        # Extract weights and biases
        w1 = state_dict[self.conv_weights[layer1_idx]]
        w2 = state_dict[self.conv_weights[layer2_idx]]

        with torch.no_grad(): 
            if (w1.shape[0] == w2.shape[1]):
                # Flip weights and biases
                state_dict[self.conv_weights[layer1_idx]][[kernel1_idx, kernel2_idx],:,:,:] = w1[[kernel2_idx, kernel1_idx],:,:,:]
                state_dict[self.conv_weights[layer2_idx]][:,[kernel1_idx, kernel2_idx],:,:] = w2[:,[kernel2_idx, kernel1_idx],:,:]
            else:
                # Flip weights and biases
                state_dict[self.conv_weights[layer1_idx]][[kernel1_idx, kernel2_idx],:,:,:] = w1[[kernel2_idx, kernel1_idx],:,:,:]
                state_dict[self.conv_weights[layer2_idx]][[kernel1_idx, kernel2_idx],:,:,:] = w2[[kernel2_idx, kernel1_idx],:,:,:]

            # Scale weights and biases
            state_dict[self.conv_weights[layer1_idx]][:,:,:,:] = w1 * scaling_factor
            state_dict[self.conv_weights[layer2_idx]][:,:,:,:] = w2 / scaling_factor
            