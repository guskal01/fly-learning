from constants import *

import random

from torch.utils.data import Dataset, DataLoader

class ShuffleAttacker():
    def __init__(self):
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

    def train_client(self, net, opt, dataset):
        train_loss = self.train(net, opt, dataset)
        self.model = net
        net = self.attack() # Maybe not necessary to assign net
        loss = self.eval_client(net, dataset)
        #print("After:", loss[-1])
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
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        return epoch_train_losses

    def eval_client(self, net, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.eval()
        batch_test_losses = []
        for data,target in dataloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())
        loss = sum(batch_test_losses)/len(batch_test_losses)

        return [loss]

    def attack(self):
        #self.shuffle_linear(0,1,0,1)

        self.shuffle_linear_random(0,1)
        self.shuffle_linear_random(1,2)
        self.shuffle_linear_random(2,3)
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