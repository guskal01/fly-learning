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
                output = server_model(data)
                loss = server_model.loss_fn(output, target)
                loss.backward()
                opt.step()

        server_state_vec = state_dict_to_vec(net.state_dict())
        server_delta = state_dict_to_vec(server_model.state_dict()) - server_state_vec

        deltas = torch.zeros((len(client_nets), len(server_state_vec)))
        for i,client_net in enumerate(client_nets):
            deltas[i] = state_dict_to_vec(client_net) - server_state_vec

        weights = torch.zeros((len(client_nets),))
        for i,delta in enumerate(deltas):
            weights[i] = max(0, F.cosine_similarity(server_delta, delta, dim=0))

        print(weights)

        result_state_vec = server_state_vec + vec_to_state_dict(torch.dot(deltas, weights)/weights.sum())

        result_state_dict = net.state_dict()
        vec_to_state_dict(result_state_vec, result_state_dict)
        net.load_state_dict(result_state_dict)

        print("testing")
        test_conversions(result_state_dict)
        print("test ok")

        return net
