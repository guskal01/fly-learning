from defences.lfr import LFR
from defences.loss_defense import LossDefense
from constants import *

import copy

class LossLFRV2():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.lfr = LFR(dataloader, n_remove=4)
        self.loss_def = LossDefense(dataloader, n_remove=4)
    
    def aggregate(self, net, client_nets, selected=None):
        lfr_net,_ = self.lfr.aggregate(copy.deepcopy(net), client_nets, selected)
        loss_net,_ = self.loss_def.aggregate(copy.deepcopy(net), client_nets, selected) 

        lfr_loss = self.get_loss(lfr_net)
        loss_loss = self.get_loss(loss_net)

        if (lfr_loss < loss_loss):
            return lfr_net, None     
        return loss_net, None

    def get_loss(self, net):
        net.eval()
        batch_test_losses = []
        for data,target in self.dataloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())
        loss = sum(batch_test_losses)/len(batch_test_losses)
        return loss
