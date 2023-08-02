from constants import *
from defences.fed_avg import FedAvg
import copy
from random import sample

class LFRBucketV2():
    def __init__(self, dataloader, n_remove):
        self.dataloader = dataloader
        self.aggregator = FedAvg(dataloader)
        self.n_remove = n_remove
        self.n_buckets = 30
        self.bucket_size = 6
    
    def aggregate(self, net, client_nets, selected):
        net_all = copy.deepcopy(self.aggregator.aggregate(net, client_nets)[0])

        best_loss = 1e9
        best_selected = None
        for bucket_itr in range(self.n_buckets):
            client_set = sample(range(len(client_nets)), self.bucket_size)

            net_selected = copy.deepcopy(self.aggregator.aggregate(net, [client_nets[i] for i in client_set])[0])
            loss = self.get_loss(net_selected)

            if (loss < best_loss):
                best_loss = loss
                best_selected = client_set

        net = copy.deepcopy(self.aggregator.aggregate(net, [client_nets[i] for i in best_selected])[0])

        return net, None

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