import torch

class ClipDefence():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def aggregate(self, net, client_nets):
        state_dict = net.state_dict()
        
        for key in state_dict:
            # Clip all weights to [-5, 5]
            state_dict[key] = sum([torch.clip(x[key], -5, 5) for x in client_nets]) / len(client_nets)
        
        net.load_state_dict(state_dict)
        return net
