from constants import *

from torch.utils.data import DataLoader

class GradiantImage():
    def __init__(self):
        pass

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for data, target in dataloader:
                data = gradient_image_attack(data)
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        return epoch_train_losses
def gradient_image_attack(image_tensor):
    gradient_image_attack = image_tensor.clone()  # create a copy of the original tensor

    # Get the dimensions of the tensor
    channels, height, width = gradient_image_attack.shape

    # Define the percentage of values to be changed (e.g., 5%)
    change_percentage = 0.05

    # Calculate the number of values to be changed
    num_changes = int(change_percentage * height * width)

    # Generate random indices for the values to be changed
    indices = torch.randint(0, height * width, (num_changes,))
    channel_indices = indices // (height * width)

    # Generate random noise for the changed values
    noise = torch.randn(num_changes) * 0.1  # adjust the strength of the noise as needed

    # Update the selected values with the noise
    for channel_idx, idx, noise_val in zip(channel_indices, indices, noise):
        gradient_image_attack[channel_idx, idx // width, idx % width] += noise_val

    # Clip the values to ensure they are within the valid range [0, 1]
    gradient_image_attack = torch.clamp(gradient_image_attack, 0, 1)

    # Return the perturbed tensor and the indices of changed values
    return gradient_image_attack, indices