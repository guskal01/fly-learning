from constants import *

from torch.utils.data import DataLoader

class RandomImageP():
    def __init__(self):
        pass

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for data, target in dataloader:
                data = random_perturbation_attack(data)
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        return epoch_train_losses
    
def random_perturbation_attack(image_tensor):
    noise = torch.randn_like(image_tensor)  # generate noise of the same shape as the image tensor
    noise = noise / torch.std(noise) * 0.1  # normalize the noise to have a certain strength (0.1 in this case)

    random_perturbation_attack = image_tensor + noise  # add the noise to the original image tensor

    # perturbed image tensor's values still fall within the valid range [0,1]
    random_perturbation_attack = torch.clamp(random_perturbation_attack, 0, 1)

    return random_perturbation_attack
