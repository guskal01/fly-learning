import torch
from defences.fed_avg import FedAvg
import copy

class NormBounding():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.aggregator = FedAvg(dataloader)
        self.tau = 0.001
        self.max_norm = 0
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()        
        for key in state_dict:
            if ("bias" in key or "weight" in key):
                for i in range(len(client_nets)):
                    client_nets[i] = self.bound_client_gradient(net, client_nets[i], key)

        net = self.aggregator.aggregate(net, client_nets)

        state_dict = net.state_dict()
        return net, None

    def bound_client_gradient(self, net, client_net, key):
        state_dict = net.state_dict()

        weight_gradient =  client_net[key] - state_dict[key]
        norm = weight_gradient.norm(p=2)

        if (norm > self.max_norm):
            self.max_norm = norm

        if (norm > self.tau):
            client_net[key] *= self.tau/norm
        
        return client_net
    
    def bound_client(self, net, client_net, key):
        state_dict = net.state_dict()

        norm = client_net[key].norm(p=2)

        if (norm > self.max_norm):
            self.max_norm = norm
            #print(self.max_norm)

        if (norm > self.tau):
            client_net[key] *= self.tau/norm
        
        return client_net

