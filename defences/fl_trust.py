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

        server_state_vec = net_to_vec(net)
        server_delta = net_to_vec(server_model) - server_state_vec

        deltas = torch.zeros((len(client_nets), len(server_state_vec))).to(device)
        for i,client_net in enumerate(client_nets):
            v = state_dict_to_vec(client_net)
            deltas[i] = v - server_state_vec
            deltas[i] *= server_delta.norm() / deltas[i].norm()

        weights = torch.zeros((len(client_nets),)).to(device)
        for i,delta in enumerate(deltas):
            weights[i] = max(0, F.cosine_similarity(server_delta, delta, dim=0))

        print(weights)

        net = weighted_avg(net, client_nets, weights)

        weights = weights.cpu().numpy()
        return net, weights/weights.sum()
