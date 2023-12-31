from constants import *
from defences.fed_avg import FedAvg
from models import Net

class LossDefense():
    def __init__(self, dataloader, n_remove):
        self.dataloader = dataloader
        self.aggregator = FedAvg(dataloader)
        self.n_remove = n_remove
    
    def aggregate(self, net, client_nets, selected):
        scores = []
        for client_idx in range(len(client_nets)):
            client_net = Net().to(device)
            client_net.load_state_dict(client_nets[client_idx])
            scores.append([self.get_loss(client_net), client_idx])

        scores.sort()
        scores = scores[::-1]

        new_nets = [client_nets[s[1]] for s in scores[self.n_remove:]]
        net,_ = self.aggregator.aggregate(net, new_nets)

        selected_clients = [s[1] for s in scores[self.n_remove:]]
        weights = [1 if i in selected_clients else 0 for i in range(len(selected))]
        print(weights)
        return net, weights

    def get_loss(self, net):
        net.eval()
        batch_test_losses = []
        for data,target in self.dataloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())
        loss = sum(batch_test_losses)/len(batch_test_losses)
        return loss