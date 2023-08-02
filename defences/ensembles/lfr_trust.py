from constants import *
from defences.fl_trust import FLTrust
from defences.fed_avg import FedAvg
from utils import *

import copy
from torch.nn import functional as F

class LFR_Trust():
    def __init__(self, dataloader, n_remove, method):
        self.dataloader = dataloader
        self.avg = FedAvg(dataloader)
        self.aggregator = FLTrust(dataloader)
        self.n_remove = n_remove
        self.method = method
    
    def aggregate(self, net, client_nets, selected):
        if (self.method == 0):
            return self.fltrust_lfr(net, client_nets, selected)
        if (self.method == 1):
            return self.lfr_fltrust(net, client_nets, selected)
        if (self.method == 2):
            return self.hybrid(net, client_nets, selected) # Not working currently (nan values after a while)

    def hybrid(self, net, client_nets, selected):
        scores = self.score_clients(net, client_nets)

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

        server_state_vec, _ = net_to_vec(net)
        server_delta = net_to_vec(server_model)[0] - server_state_vec

        deltas = torch.zeros((len(client_nets), len(server_state_vec))).to(device)
        for i,client_net in enumerate(client_nets):
            v, nbt = state_dict_to_vec(client_net)
            deltas[i] = v - server_state_vec
            deltas[i] *= server_delta.norm() / deltas[i].norm()

        weights = torch.zeros((len(client_nets),)).to(device)
        for i,delta in enumerate(deltas):
            weights[i] = max(0, F.cosine_similarity(server_delta, delta, dim=0))

        print(f"Before: {weights}")
        for s in scores[:self.n_remove]:
            weights[s[1]] = 0
        print(f"After: {weights}")

        result_state_vec = server_state_vec + ((deltas*weights.unsqueeze(1))/weights.sum()).sum(dim=0)

        result_state_dict = net.state_dict()
        vec_to_state_dict(result_state_vec, result_state_dict, nbt)
        net.load_state_dict(result_state_dict)

        weights = weights.cpu().numpy()
        return net, weights/weights.sum()

    def fltrust_lfr(self, net, client_nets, selected):
        scores = self.score_clients(net, client_nets)
        print("Removed:", [selected[s[1]] for s in scores[:self.n_remove]])

        # FLTrust first, then LFR
        _, weights = self.aggregator.aggregate(net, client_nets, None)
        for s in scores[:self.n_remove]:
            weights[s[1]] = 0
        weights /= weights.sum()
        net = weighted_avg(net, client_nets, weights)
        net = copy.deepcopy(net)

        return net, weights

    def lfr_fltrust(self, net, client_nets, selected):
        scores = self.score_clients(net, client_nets)
        print("Removed:", [selected[s[1]] for s in scores[:self.n_remove]])

        # LFR first, then FLTrust
        new_nets = [client_nets[s[1]] for s in scores[self.n_remove:]]
        net, weights = self.aggregator.aggregate(net, new_nets, None)
        net = copy.deepcopy(net)

        # selected_clients = [s[1] for s in scores[self.n_remove:]]
        # weights = [weights[selected_clients.index(scores[i][1])] if scores[i][1] in selected_clients else 0 for i in range(len(selected))]

        return net, None #weights

    def score_clients(self, net, client_nets):
        org_net = copy.deepcopy(net)
        net_all = copy.deepcopy(self.avg.aggregate(net, client_nets)[0])

        scores = []
        for client_idx in range(len(client_nets)):
            net_without_client = copy.deepcopy(self.avg.aggregate(net, client_nets[:client_idx]+client_nets[client_idx+1:])[0])
            scores.append([self.loss_impact(net_all, net_without_client), client_idx])

        scores.sort()
        scores = scores[::-1]

        net = copy.deepcopy(org_net)
        return scores

    def loss_impact(self, net_all, net_without_client):
        l_all = self.get_loss(net_all)
        l_without = self.get_loss(net_without_client)
        return l_all-l_without

    def get_loss(self, net):
        net.eval()
        batch_test_losses = []
        for data,target in self.dataloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())
        loss = sum(batch_test_losses)/len(batch_test_losses)
        return loss