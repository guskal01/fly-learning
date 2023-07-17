from constants import *
from models import Net
from utils import *

from torch.nn import functional as F

class FLTrust():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def aggregate(self, net, client_nets, selected):
        server_model = Net().to(device)
        server_model.load_state_dict(net.state_dict())
        server_model.train()
        opt = torch.optim.Adam(server_model.parameters(), lr=0.001)
        for epoch in range(1, 1+1):
            for data, target in self.dataloader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()
        
        old_server_state_vec = state_dict_to_vec(net.state_dict())
        server_state_vec = state_dict_to_vec(server_model.state_dict())

        weights = []
        for client_net in client_nets:
            state_vec = state_dict_to_vec(client_net)
            weights.append(max(0, F.cosine_similarity(server_state_vec-old_server_state_vec, state_vec-old_server_state_vec, dim=0)))

        state_dict = net.state_dict()
        
        for key in state_dict:
            state_dict[key] = sum([x[key]*w for x,w in zip(client_nets, weights)]) / sum(weights)
        
        net.load_state_dict(state_dict)
        return net
