from ..constants import *
from fed_avg import FedAvg

class LFR():
    def __init__(self, dataloader, n_remove):
        self.dataloader = dataloader
        self.aggregator = FedAvg(dataloader)
        self.n_remove = n_remove
    
    def aggregate(self, net, client_nets):
        net_all = self.aggregator.aggregate(net, client_nets)

        scores = []
        for client_idx in range(len(client_nets)):
            net_without_client = self.aggregate(net, client_nets[:client_idx]+client_nets[client_idx+1:])
            scores.append([self.loss_impact(net_all, net_without_client), client_idx])

        scores.sort()[::-1]

        new_nets = [s[1] for s in scores[self.n_remove:]]
        net = self.aggregate(net, new_nets)
        return net

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