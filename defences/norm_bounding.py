import torch
from defences.fed_avg import FedAvg
import copy

class NormBounding():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.aggregator = FedAvg(dataloader)
        self.tau =  1.0
        self.max_norm = 0
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()        
        for key in state_dict:
            if 'num_batches_tracked' in key:
                state_dict[key] = client_nets[0][key]
            else:
                for i in range(len(client_nets)):
                    client_nets[i] = self.bound_client_gradient(net, client_nets[i], key)

        net.load_state_dict(state_dict)
        net, _ = self.aggregator.aggregate(net, client_nets)
        net.load_state_dict(state_dict)
        return net, None

    def bound_client_gradient(self, net, client_net, key):
        state_dict = net.state_dict()

        weight_gradient =  client_net[key] - state_dict[key]
        norm = weight_gradient.norm(p=2)

        if (norm > self.max_norm):
            self.max_norm = norm
            print(norm)

        if (norm > self.tau):
            client_net[key] = state_dict[key] + weight_gradient * self.tau/norm
        
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

