from constants import *

from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np

class NeurotoxinAttack():
    def __init__(self):
        self.grad = []
        self.grad_list = {}
        self.curr_round = 0
        self.sorted_indices= {}
        self.least_k_percent=0
        self.least_k_percent_indices=[]
        self.sorted_values=[]
        self.model_param=0
        self.net_copy={}
    
    def train_client(self, net, opt, dataset):
        self.curr_round += 1
        dataloader = DataLoader(dataset, batch_size=32)
       
        if(self.curr_round <3):
            if(self.curr_round==2):
                for key in net.state_dict():
                    if ("weight" in key or "bias" in key):
                        # net.load_state_dict(net_try_copy)
                        # net_copy.params= net_copy.state_dict()
                        self.grad_list[key]=net.state_dict()[key]-self.net_copy.state_dict()[key]
                        # if(self.curr_round < 1): return [420]
                        self.sorted_values, self.sorted_indices = torch.sort(self.grad_list[key], descending=False)
                        self.least_k_percent= int(0.10 * len(self.sorted_indices))
                        self.least_k_percent_indices= self.sorted_indices[:self.least_k_percent]
                        self.sorted_values.zero_()
                        self.grad_list[self.least_k_percent_indices]=self.sorted_values[:self.least_k_percent]
                        net.state_dict()[key] += self.grad_list[key]
            
            net.train()
            
            for epoch in range(1, EPOCHS_PER_ROUND+1):
                batch_train_losses = []
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    opt.zero_grad()
                    output = net(data)
                    loss = net.loss_fn(output, target)
                    loss.backward()
                    opt.step()
            else:
                self.net_copy= copy.deepcopy(net)
        else:
            print("else loop")
            net.load_state_dict(net.state_dict())
            net.train()
            for epoch in range(1, EPOCHS_PER_ROUND+1):
                batch_train_losses = []
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    opt.zero_grad()
                    output = net(data)
                    loss = net.loss_fn(output, target)
                    loss.backward()
                    opt.step()
        return [420]




 
 