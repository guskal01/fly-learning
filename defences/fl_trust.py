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
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            for data, target in self.dataloader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = server_model(data)
                loss = server_model.loss_fn(output, target)
                loss.backward()
                opt.step()

        server_state_vec = net_to_params(net)
        server_delta = net_to_params(server_model) - server_state_vec

        deltas = torch.zeros((len(client_nets), len(server_state_vec))).to(device)
        for i,client_net in enumerate(client_nets):
            cn = Net().to(device)
            cn.load_state_dict(client_net)
            deltas[i] = net_to_params(cn) - server_state_vec
            deltas[i] *= server_delta.norm() / deltas[i].norm()

        weights = torch.zeros((len(client_nets),)).to(device)
        for i,delta in enumerate(deltas):
            weights[i] = max(0, F.cosine_similarity(server_delta, delta, dim=0))

        print(weights)

        state_dict = net.state_dict()
        for key in state_dict:
            state_dict[key] = sum([x[key]*weights[i] for i,x in enumerate(client_nets)]) / weights.sum()
        net.load_state_dict(state_dict)
        return net
