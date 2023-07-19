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


    def find_gradients(self,net):
        net_copy= copy.deepcopy(net)
        for key in net.state_dict():
            if ("weight" in key or "bias" in key):
                self.grad_list[key]=net.state_dict()[key]-net_copy.state_dict()[key]
                # self.grad_list[key]=net.state_dict()[key]-net.load_state_dict(net.state_dict()[key])
                print("000000")
                if (self.curr_round < 1): return [420] 
                self.sorted_values, self.sorted_indices = torch.sort(self.grad_list[key], descending=False)
                print(" between 0 and 1")
                self.least_k_percent= int(0.10 * len(self.sorted_indices))
                self.least_k_percent_indices= self.sorted_indices[:self.least_k_percent]
                print("1111111")
                self.sorted_values.zero_()
                print("222222")
                self.grad_list[self.least_k_percent_indices]=self.sorted_values[:self.least_k_percent]
                print("3333333")
                net.state_dict()[key] += self.grad_list[key]
                print("444444")
                # print("end of find grad function")

    def train_client(self, net, opt, dataset):
        self.curr_round += 1
        # net_copy={}
        # if (self.curr_round ==1):
        #         # net_copy= net.load_state_dict(net.state_dict())
        #         net_copy= net.state_dict()
        #         return [420]
        
        
        dataloader = DataLoader(dataset, batch_size=32)
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
        for key in net.state_dict():
            if (self.curr_round <1):
                # net.load_state_dict(net.state_dict())
                return [420]
            if (self.curr_round == 2):
                self.find_gradients(net)
                print("got grad")
                for key in net.state_dict():
                    if ("weight" in key or "bias" in key):
                        net.state_dict()[key] += self.grad_list[key]

            else:
                print("else loop")
                net.load_state_dict(net.state_dict())
            # if ("weight" in key or "bias" in key):
            #     self.grad_list[key]=net.state_dict()[key]-net_copy.state_dict()[key]
            #     if (self.curr_round < 1): return [420]
            #     self.sorted_values, self.sorted_indices = torch.sort(self.grad_list[key], descending=False)
            #     self.least_k_percent= int(0.10 * len(self.sorted_indices))
            #     self.least_k_percent_indices= self.sorted_indices[:self.least_k_percent]
            #     self.sorted_values.zero_()
            #     self.grad_list[self.least_k_percent_indices]=self.sorted_values[:self.least_k_percent]


            #     net.state_dict()[key] += self.grad_list[key]
        return [420]
 