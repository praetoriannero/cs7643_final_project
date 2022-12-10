import numpy
import random
from tqdm import tqdm
import torch
from torch.utils.data import Subset
import torchvision

from ensemble_ad.utils import get_cifar10_dataset


class ResnetOutlierExposure(torch.nn.Module):
    def __init__(
            self,
            anomaly_labels: list,
            num_classes=10,
            lr : float = 1e-3,
            batch_size : int = 128,
            epochs : int = 10,
            lambda_hp : float = 0.5,
            device : str = "cpu") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anomaly_labels = torch.tensor(anomaly_labels)
        self.lambda_hp = lambda_hp
        self.epochs = epochs
        self.device = device

        self.epoch_losses = []
        self.batch_losses = []
        self.batch_accuracy = []
        self.epoch_accuracy = []
        self.test_losses = []

        self.model = torchvision.models.convnext_tiny(num_classes=num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # if self.num_classes == 2:
        #     self.ce_loss = torch.nn.BCELoss()
        # else:
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.anomaly_labels = self.anomaly_labels.to(self.device)
        self.model = self.model.to(self.device)
        # self.optimizer = self.optimizer.to(self.device)
        self.ce_loss = self.ce_loss.to(self.device)
        self.to(self.device)

    def train(self, train_dataloader, test_dataloader):
        for epoch in range(self.epochs):
            print(f"\nEpoch: {epoch + 1}/{self.epochs}")
            self.model.train()
            avg_loss = 0
            avg_accuracy = 0
            for inputs, labels in tqdm(train_dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                in_dist_indices = torch.isin(labels, self.anomaly_labels, invert=True)
                out_dist_indices = torch.isin(labels, self.anomaly_labels)

                outputs = self.model(inputs)

                if self.num_classes == 2:
                    binary_labels = out_dist_indices.long()
                    labels = binary_labels

                in_dist_outputs = outputs[in_dist_indices]
                # in_dist_labels = in_dist_indices.float()
                in_dist_labels = labels[in_dist_indices]

                out_dist_outputs = outputs[out_dist_indices]

                self.optimizer.zero_grad()

                # calc loss here
                ce_loss = self.ce_loss(in_dist_outputs, in_dist_labels)

                values, predictions = in_dist_outputs.max(1)
                batch_accuracy = predictions.eq(in_dist_labels.data).sum().item()
                # input()
                avg_accuracy += batch_accuracy
                self.batch_accuracy.append(float(batch_accuracy))

                # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
                oe_loss = self.lambda_hp * -(out_dist_outputs.mean(1) - torch.logsumexp(
                    out_dist_outputs, dim=1)).mean()
                # print(oe_loss)
                # input()

                loss = ce_loss #+ oe_loss
                loss.backward()

                self.optimizer.step()

                avg_loss += float(loss)
                self.batch_losses.append(float(loss))

            self.epoch_losses.append(avg_loss / len(train_dataloader))
            self.epoch_accuracy.append(float(avg_accuracy) / len(train_dataloader.dataset))
            print(f"Train loss: {round(self.epoch_losses[-1], 6)}")
            print(f"Train acc: {round(self.epoch_accuracy[-1] * 100, 6)}%")

            with torch.no_grad():
                self.model.eval()
                dev_avg_loss = 0
                valid_correct = 0
                for inputs, labels in tqdm(test_dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    in_dist_indices = torch.isin(labels, self.anomaly_labels, invert=True)
                    out_dist_indices = torch.isin(labels, self.anomaly_labels)

                    outputs = self.model(inputs)

                    if self.num_classes == 2:
                        binary_labels = out_dist_indices.long()
                        labels = binary_labels
                        # outputs = outputs.max(1)

                    in_dist_outputs = outputs[[in_dist_indices.squeeze()]]
                    in_dist_labels = labels[[in_dist_indices.squeeze()]]

                    out_dist_outputs = outputs[[out_dist_indices.squeeze()]]
                    ce_loss = self.ce_loss(in_dist_outputs, in_dist_labels)

                    # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
                    oe_loss = self.lambda_hp * -(
                        out_dist_outputs.mean(1) - torch.logsumexp(out_dist_outputs, dim=1)
                    ).mean()

                    loss = ce_loss #+ oe_loss
                    dev_avg_loss += float(loss)

                    values, predictions = in_dist_outputs.max(1)
                    valid_correct += predictions.eq(in_dist_labels.data).sum().item()

                test_loss = dev_avg_loss / len(test_dataloader)
                self.test_losses.append(test_loss)
                print(f"Valid loss: {round(test_loss, 6)}")
                print(f"Valid acc: {round(valid_correct / len(test_dataloader.dataset) * 100, 6)}%")

    def eval(self, eval_dataloader):
        pass


def main():
    seed = 42
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

    wide_resnet_oe = ResnetOutlierExposure(
        anomaly_labels,
        num_classes=10,
        epochs=20,
        lr=0.1,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available else "cpu")
    wide_resnet_oe.train(train_dl, test_dl)

    import matplotlib.pyplot as plt

    plt.plot(wide_resnet_oe.epoch_losses)
    plt.plot(wide_resnet_oe.test_losses)
    plt.show()


if __name__ == "__main__":
    main()
