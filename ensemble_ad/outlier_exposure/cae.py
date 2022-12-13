import numpy as np
import random
import torch
from torch import nn
from torchvision.utils import save_image
from torchmetrics.functional.classification import binary_auroc, binary_accuracy
from tqdm import tqdm

from utils import get_cifar10_dataset, get_mnist_dataset, get_svhn_dataset

# From https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
class EncoderBlock(nn.Module):
    def __init__(self, channels, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 8, 3),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU()
        )
        self.model.to(device)
        self.to(device)

    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, channels, 3)
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
            nn.Linear(16 * 24 * 24, latent_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 32 * 26 * 26),
            nn.Unflatten(1, torch.Size([32, 26, 26])),
            nn.LeakyReLU()
        )
        self.model.to(device)
        self.to(device)

    def forward(self, x):
        # print(x.shape)
        return self.model(x)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_size, channels, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            EncoderBlock(channels, device),
            LatentBlock(latent_size, device),
            DecoderBlock(channels, device)
        )
        self.rl_min = 0
        self.rl_max = 1
        self.loss_function = nn.MSELoss().to(device)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        return torch.sigmoid(self.model(x))

    def predict(self, x):
        x_r = self.forward(x)
        loss = self.loss_function(x_r, x)
        # return loss
        normalized_loss = torch.abs((loss - self.rl_min) / (self.rl_max - self.rl_min))
        return torch.sigmoid(normalized_loss)


def train(train_dataloader, model, in_dist_labels, loss_function, optimizer, epoch, device='cpu'):
    model.train()
    total_loss = 0
    in_dist_labels = torch.tensor(in_dist_labels).to(device)
    for inputs, labels in tqdm(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if len(in_dist_labels) > 0:
            in_dist_indices = torch.isin(labels, in_dist_labels)
            # print(inputs.shape)
            inputs = inputs[in_dist_indices]
            # print(inputs.shape)
            # exit()

        outputs = model(inputs)

        # print(outputs[0])
        # print(inputs[0])
        # input()

        loss = loss_function(outputs, inputs)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return model, total_loss


def test(test_dataloader, model, in_dist_labels, loss_function, device='cpu'):
    model.eval()
    total_loss = 0
    in_dist_labels = torch.tensor(in_dist_labels).to(device)
    max_rl = -1e9
    min_rl = 1e9
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if len(in_dist_labels) > 0:
                in_dist_indices = torch.isin(labels, in_dist_labels)
                # print(inputs.shape)
                inputs = inputs[in_dist_indices]
                # print(inputs.shape)

            outputs = model(inputs)
            loss = loss_function(outputs, inputs)

            fl = loss.item()

            if fl > max_rl:
                max_rl = fl
            if fl < min_rl:
                min_rl = fl

            total_loss += fl

    model.rl_max = max_rl
    model.rl_min = min_rl
    return model, total_loss


def get_anomaly_scores(test_dataloader, model, device, in_dist_labels):
    model.eval()
    label_scores = None
    # total_predictions = 0
    label_scores = [0 for i in range(10)]
    label_count = [0 for i in range(10)]

    true_positives = []
    true_negatives = []

    false_positives = []
    false_negatives = []

    preds = []
    truth = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            for idx, x_in in enumerate(inputs):
                score = model.predict(x_in.unsqueeze(0))
                preds.append(score.item())
                truth.append(0 if labels[idx].data in in_dist_labels else 1)

                label_scores[labels[idx].data] += score.item()
                label_count[labels[idx].data] += 1

    bin_auroc = binary_auroc(torch.tensor(preds), torch.tensor(truth))
    print(f"AUROC: {bin_auroc.item()}")

    label_scores = np.array(label_scores)
    label_count = np.array(label_count)

    print(label_scores / label_count)

def main():
    seed = 42
    device = "cuda" if torch.cuda.is_available else "cpu"
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    batch_size = 256

    dataset = "cifar10"

    if dataset == "mnist":
        mnist_dataset = get_mnist_dataset()
        train_idx, test_idx = torch.utils.data.random_split(
            mnist_dataset,
            [0.8, 0.2]
        )
        channels = 1


    elif dataset == "cifar10":
        cifar10_dataset = get_cifar10_dataset()
        train_idx, test_idx = torch.utils.data.random_split(
            cifar10_dataset,
            [0.8, 0.2]
        )
        channels = 3

    elif dataset == "svhn":
        svhn_dataset = get_svhn_dataset()
        train_idx, test_idx = torch.utils.data.random_split(
            svhn_dataset,
            [0.8, 0.2]
        )
        channels = 3

    train_dl = torch.utils.data.DataLoader(
        train_idx, batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(
        test_idx, batch_size=batch_size)

    latent_size = 64


    for i in range(10):
        i = 1
        cae = ConvolutionalAutoencoder(
            latent_size,
            channels,
            device=device
        )
        lr = 1e-3
        epochs = 10
        optimizer = torch.optim.Adam(cae.parameters(), lr)
        loss_function = nn.MSELoss().to(device)
        train_losses = []
        test_losses = []
        for inputs, labels in test_dl:
            test_image = torch.unsqueeze(inputs[0].to(device), dim=0)
            test_label = labels[0]

        in_dist_labels = [i]
        for epoch in range(epochs):
            print(f"\nEpoch: {epoch + 1}/{epochs}")

            cae, train_loss = train(
                train_dl, cae, in_dist_labels, loss_function, optimizer, epoch, device=device)

            cae, test_loss = test(
                test_dl, cae, in_dist_labels, loss_function, device=device)

            train_losses.append(train_loss / len(train_dl))
            test_losses.append(test_loss / len(test_dl))

            print(f"Train loss: {round(train_losses[-1], 6)}")
            print(f"Test loss: {round(test_losses[-1], 6)}")

            rec_output = cae(test_image)
            combined = torch.cat((test_image, rec_output), 2)

            random_cae_img = save_image(combined, f"CAE_class_{test_label}_EPOCH_{epoch}.png")

        print("\nValidation Test")
        get_anomaly_scores(test_dl, cae, device, in_dist_labels)

        model_name = f"CAE_model_class_{i}_{latent_size}_{lr}_{epochs}"
        torch.save(cae, f"{model_name}.pth")

        import matplotlib.pyplot as plt

        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.savefig(f"CAE_model_class_{i}_{latent_size}_{lr}_{epochs}_history.png")
        break


if __name__ == "__main__":
    main()


# [0.01841423 0.01918637 0.01556432 0.01720019 0.01419787 0.01790166
#  0.01630134 0.01880272 0.01527021 0.02008243]
