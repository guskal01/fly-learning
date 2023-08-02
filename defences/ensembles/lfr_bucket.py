from constants import *
from defences.fed_avg import FedAvg
import copy
from random import sample

class LFRBucket():
    def __init__(self, dataloader, n_remove):
        self.dataloader = dataloader
        self.aggregator = FedAvg(dataloader)
        self.n_remove = n_remove
        self.n_buckets = 30
        self.bucket_size = 6
    
    def aggregate(self, net, client_nets, selected):
        net_all = copy.deepcopy(self.aggregator.aggregate(net, client_nets)[0])

        scores = [[0,0,0] for _ in range(len(client_nets))]
        for bucket_itr in range(self.n_buckets):
            client_set = sample(range(len(client_nets)), self.bucket_size)

            net_without_client = copy.deepcopy(self.aggregator.aggregate(net, [client_nets[i] for i in client_set])[0])
            impact = self.loss_impact(net_all, net_without_client)

            for client_idx in range(len(client_nets)):
                if (client_idx not in client_set):
                    scores[client_idx] = [scores[client_idx][0]+impact, client_idx, scores[client_idx][2]+1]

        print(f"Selected: {selected}")
        print(scores)
        scores = [[s[0]/s[2],s[1]] if s[2] != 0 else [0,s[1]] for s in scores]
        
        scores.sort()
        scores = scores[::-1]
        print(scores)

        print("Removed:", [selected[s[1]] for s in scores[:self.n_remove]])

        new_nets = [client_nets[s[1]] for s in scores[self.n_remove:]]
        net = copy.deepcopy(self.aggregator.aggregate(net, new_nets)[0])

        selected_clients = [s[1] for s in scores[self.n_remove:]]
        weights = [1 if i in selected_clients else 0 for i in range(len(selected))]
        return net, weights

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