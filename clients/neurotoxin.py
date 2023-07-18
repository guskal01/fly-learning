from constants import *

from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np

class NeurotoxinAttack():
    def __init__(self):
        self.grad = []
        self.grad_list = {}
        self.grad_try=[]
        self.grad_try_prev=[]
        self.curr_round = 0
        self.sorted_indices= {}
        self.least_k_percent=0
        self.least_k_percent_indices=[]
        self.sorted_values=[]
        

    def train_client(self, net, opt, dataset):
        self.curr_round += 1
        
        dataloader = DataLoader(dataset, batch_size=32)
        net_copy= copy.deepcopy(net)
        # net_copy = net.deepcopy()
        # net_copy.train()
        # epoch_train_losses = []
        # for epoch in range(1, EPOCHS_PER_ROUND+1):
        #     batch_train_losses = []
        #     for data, target in dataloader:
        #         data, target = data.to(device), target.to(device)
        #         opt.zero_grad()
        #         output = net_copy(data)
        #         loss = net.loss_fn(output, target)
        #         loss.backward()
        #         opt.step()
                
       
        net.train()
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                # for i in net.named_parameters():
                #     self.grad_try=net.named_parameters()[i]
                #     # list_grad_try=list(self.grad_try)
                #     print("self.grad_try",self.grad_try)
                #     # print("len of self.grad_try",len(self.grad_try))
                # for j in net_copy.named_parameters():
                #     self.grad_try_prev= net_copy.named_parameters()[j]
                #     # list_grad_try_prev=self.grad_try_prev
                #     print("self.grad_try_prev",self.grad_try_prev)
                    # print("len of self.grad_try_prev", len(list_grad_try_prev))
                # self.grad_list= np.subtract(self.grad_try,self.grad_try_prev)
                # print("self grad list", self.grad_list)
                # l=net.state_dict()
                # print("weight",l['weight'])
                
                #     self.grad = net.state_dict()[key] - net_copy.state_dict()[key]
                #     #print("gradients", self.grad)
            
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()
                # for key in net.state_dict():
                #     if ("weight" in key or "bias" in key):
                #         # print("new model weights",net.state_dict()[key])
                #         # print("prev model weights",net_copy.state_dict()[key] )
                #         self.grad_list[key]=net.state_dict()[key]-net_copy.state_dict()[key]
                #         print("key",self.grad_list)
        for key in net.state_dict():
            if ("weight" in key or "bias" in key):
                # print("new model weights",net.state_dict()[key])
                # print("prev model weights",net_copy.state_dict()[key] )
                
                self.grad_list[key]=net.state_dict()[key]-net_copy.state_dict()[key]
                if (self.curr_round < 2): return [420] 
                self.sorted_values, self.sorted_indices = torch.sort(self.grad_list[key], descending=False)
                # print("key",self.grad_list)
                # print("len of weight", len(self.grad_list[key]))
                # print("sorted weights indices",0.10 * len(self.sorted_indices))
                self.least_k_percent= int(0.10 * len(self.sorted_indices))
                self.least_k_percent_indices= self.sorted_indices[:self.least_k_percent]
                # print("self.least_k_percent_indices",self.least_k_percent_indices)
                self.sorted_values.zero_()
                self.grad_list[self.least_k_percent_indices]=self.sorted_values[:self.least_k_percent]
                print("updated sorted indices",self.grad_list)
        # self.grad_list.append(self.grad)
        # print("gradient list", len(self.grad_list))
        return [420]
 