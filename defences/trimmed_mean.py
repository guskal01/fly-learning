import torch
import numpy as np
import copy

class TrimmedMean():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.k = 2
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()        
        for key in state_dict:
            state_dict[key] = self.trim_mean(client_nets, key)

        net.load_state_dict(state_dict)
        return net, None

    def trim_mean(self, client_nets, key):
        client_params = torch.from_numpy(np.array([client_nets[i][key].cpu().numpy().flatten() for i in range(len(client_nets))]))
        sorted_params = client_params.sort(dim=0)[0]
        trimmed_params = sorted_params[self.k:-self.k]
        return trimmed_params.float().mean(dim=0).reshape(client_nets[0][key].shape)
