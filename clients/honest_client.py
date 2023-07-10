from constants import *

from torch.utils.data import Dataset, DataLoader

class HonestClient():
    def __init__(self):
        pass

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            print("Epoch", epoch)
            batch_train_losses = []
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        return epoch_train_losses
