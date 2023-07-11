from constants import *

from torch.utils.data import Dataset, DataLoader

class NoTrainClient():
    def __init__(self):
        pass

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.eval()
        batch_test_losses = []
        for data,target in dataloader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            batch_test_losses.append(net.loss_fn(pred, target).item())
        loss = sum(batch_test_losses)/len(batch_test_losses)

        return [loss]
