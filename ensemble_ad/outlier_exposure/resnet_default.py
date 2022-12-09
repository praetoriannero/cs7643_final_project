from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torchmetrics.classification import BinaryAUROC
import torchvision

from ensemble_ad.utils import get_cifar10_dataset


class WideResnet(torch.nn.Module):
    def __init__(
            self,
            anomaly_labels: list,
            lr : float = 1e-3,
            batch_size : int = 128,
            epochs : int = 10,
            device : str = "cpu") -> None:
        super().__init__()
        self.anomaly_labels = torch.tensor(anomaly_labels)
        self.epochs = epochs
        self.device = device

        self.epoch_losses = []
        self.batch_losses = []
        self.batch_accuracy = []
        self.epoch_accuracy = []
        self.test_losses = []

        self.model = torchvision.models.wide_resnet50_2()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.auroc = BinaryAUROC()

        self.anomaly_labels = self.anomaly_labels.to(self.device)
        self.model = self.model.to(self.device)
        self.ce_loss = self.ce_loss.to(self.device)
        self.auroc = self.auroc.to(self.device)
        self.to(self.device)

    def train(self, train_dataloader, dev_dataloader):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}")
            self.model.train()
            avg_loss = 0
            avg_accuracy = 0
            for inputs, labels in tqdm(train_dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                binary_labels = torch.isin(labels, self.anomaly_labels).long()

                outputs = self.model(inputs)

                self.optimizer.zero_grad()

                # calc loss here
                loss = self.ce_loss(outputs, binary_labels)
                predicted_labels = torch.argmax(outputs, dim=1)

                batch_accuracy = torch.sum(predicted_labels == binary_labels) / len(binary_labels)
                avg_accuracy += batch_accuracy
                self.batch_accuracy.append(float(batch_accuracy))

                loss.backward()

                self.optimizer.step()

                avg_loss += float(loss)
                self.batch_losses.append(float(loss))

            self.epoch_losses.append(avg_loss / len(train_dataloader))
            self.epoch_accuracy.append(float(avg_accuracy) / len(train_dataloader))
            print(f"Current epoch training loss: {self.epoch_losses[-1]}")
            print(f"Current epoch accuracy: {self.epoch_accuracy[-1] * 100}%")

            with torch.no_grad():
                self.model.eval()
                dev_avg_loss = 0
                for inputs, labels in tqdm(dev_dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    binary_labels = torch.isin(labels, self.anomaly_labels).long()

                    outputs = self.model(inputs)

                    self.optimizer.zero_grad()

                    # calc loss here
                    loss = self.ce_loss(outputs, binary_labels)
                    predicted_labels = torch.argmax(outputs, dim=1)

                    dev_avg_loss += float(loss)

                    valid_accuracy = torch.sum(predicted_labels == binary_labels) / len(binary_labels)

                print(f"Validation loss: {dev_avg_loss / len(dev_dataloader)}")
                print(f"Validation acc: {valid_accuracy * 100}%")

    def eval(self, eval_dataloader):
        pass


def main():
    torch.random.manual_seed(42)

    batch_size = 128

    cifar10_dataset = get_cifar10_dataset()

    train_idx, test_idx, dev_idx = torch.utils.data.random_split(
        cifar10_dataset,
        [0.8, 0.1, 0.1]
    )

    anomaly_labels = [6, 7, 8, 9]
    invalid = torch.tensor(anomaly_labels)

    train_dl = torch.utils.data.DataLoader(
        train_idx, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_idx, batch_size=batch_size, shuffle=True)
    dev_dl = torch.utils.data.DataLoader(
        dev_idx, batch_size=batch_size, shuffle=True)

    wide_resnet = WideResnet(
        anomaly_labels,
        epochs=10,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available else "cpu")
    wide_resnet.train(train_dl, dev_dl)

    import matplotlib.pyplot as plt

    plt.plot(wide_resnet.epoch_accuracy)
    plt.show()


if __name__ == "__main__":
    main()
