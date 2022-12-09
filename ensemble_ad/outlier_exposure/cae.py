import numpy
import random
import torch
from torch import nn
from tqdm import tqdm

from ensemble_ad.utils import get_cifar10_dataset

# From https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
class EncoderBlock(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU()
        )
        self.model.to(device)
        self.to(device)

    def forward(self, x):
        # print(x.type())
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3)
        )
        self.model.to(device)
        self.to(device)

    def forward(self, x):
        return self.model(x)


class LatentBlock(nn.Module):
    def __init__(self, latent_size=4, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 32 * 32 * 3),
            nn.Unflatten(1, torch.Size([3, 32, 32])),
            nn.ReLU()
        )
        self.model.to(device)
        self.to(device)

    def forward(self, x):
        # print(x.shape)
        return self.model(x)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_size, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            EncoderBlock(device),
            DecoderBlock(device),
            LatentBlock(latent_size, device)
        )
        self.loss_function = nn.MSELoss().to(device)

        self.to(device)

    def forward(self, x):
        return self.model(x)

    def get_mse(self, x, y):
        x_r = self.forward(x)

        return self.loss_function(x_r, y)


def train(train_dataloader, model, anomaly_labels, loss_function, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_function(outputs, inputs)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        total_loss += float(loss)


    return model, total_loss


def test(test_dataloader, model, anomaly_labels, loss_function, device='cpu'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, inputs)

            total_loss += float(loss)


    return model, total_loss


def main():
    seed = 42
    device = "cuda" if torch.cuda.is_available else "cpu"
    torch.random.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    batch_size = 128

    cifar10_dataset = get_cifar10_dataset()

    train_idx, test_idx = torch.utils.data.random_split(
        cifar10_dataset,
        [0.8, 0.2]
    )

    anomaly_labels = []
    invalid = torch.tensor(anomaly_labels)

    train_dl = torch.utils.data.DataLoader(
        train_idx, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_idx, batch_size=batch_size, shuffle=True)

    cae = ConvolutionalAutoencoder(
        latent_size=8,
        device=device
    )
    lr = 1e-3
    epochs = 10
    optimizer = torch.optim.Adam(cae.parameters(), lr)
    loss_function = nn.MSELoss().to(device)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")

        cae, train_loss = train(
            train_dl, cae, anomaly_labels, loss_function, optimizer, device=device)

        cae, test_loss = test(
            test_dl, cae, anomaly_labels, loss_function, device=device)

        train_losses.append(train_loss / len(train_dl))
        test_losses.append(test_loss / len(test_dl))

        print(f"Train loss: {round(train_losses[-1], 6)}")
        print(f"Test loss: {round(test_losses[-1], 6)}")

    import matplotlib.pyplot as plt

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()


if __name__ == "__main__":
    main()
