import torch
import numpy as np
import copy

class TrimmedMean():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.k = 1
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()        
        for key in state_dict:
            if ("bias" in key or "weight" in key):
                state_dict[key] = self.trim_mean(client_nets, key)

        state_dict = net.state_dict()
        return net

    def trim_mean(self, client_nets, key):
        client_params = torch.from_numpy(np.array([client_nets[i][key].cpu().numpy() for i in range(len(client_nets))]))
        sorted_params = client_params.sort(dim=0)[0]
        trimmed_params = sorted_params[self.k:-self.k]
        return trimmed_params.mean(dim=0)
