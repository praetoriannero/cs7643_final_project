from tqdm import tqdm
import torch
from torch.utils.data import Subset
import torchvision

from ensemble_ad.utils import get_cifar10_dataset


class WideResnetOutlierExposure(torch.nn.Module):
    def __init__(
            self,
            anomaly_labels: int,
            lr : float = 1e-3,
            batch_size : int = 128,
            epochs : int = 10,
            lambda_hp : float = 0.5,
            device : str = "cpu") -> None:
        super().__init__()
        self.anomaly_labels = torch.tensor(anomaly_labels)
        self.lambda_hp = lambda_hp
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

        self.anomaly_labels = self.anomaly_labels.to(self.device)
        self.model = self.model.to(self.device)
        # self.optimizer = self.optimizer.to(self.device)
        self.ce_loss = self.ce_loss.to(self.device)
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

                in_dist_indices = torch.isin(labels, self.anomaly_labels)
                out_dist_indices = torch.isin(labels, self.anomaly_labels, invert=True)
                # out_dist_outputs = labels[in_dist_indices == False]

                outputs = self.model(inputs)

                in_dist_outputs = outputs[[in_dist_indices.squeeze()]]
                # in_dist_labels = in_dist_indices.float()
                in_dist_labels = labels[[in_dist_indices.squeeze()]]

                out_dist_outputs = outputs[[out_dist_indices.squeeze()]]

                self.optimizer.zero_grad()

                # calc loss here
                ce_loss = self.ce_loss(in_dist_outputs, in_dist_labels)
                predicted_labels = torch.argmax(in_dist_outputs, dim=1)
                actual_labels = in_dist_labels
                # batch_accuracy = predicted_labels == actual_labels
                batch_accuracy = torch.sum(predicted_labels == actual_labels) / len(actual_labels)
                avg_accuracy += batch_accuracy
                self.batch_accuracy.append(float(batch_accuracy))

                # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
                oe_loss = self.lambda_hp * -(
                    out_dist_outputs.mean(1) - torch.logsumexp(out_dist_outputs, dim=1)
                ).mean()

                loss = ce_loss + oe_loss
                loss.backward()

                self.optimizer.step()

                avg_loss += float(loss)
                self.batch_losses.append(float(loss))

            self.epoch_losses.append(avg_loss / len(train_dataloader))
            self.epoch_accuracy.append(float(batch_accuracy) / len(train_dataloader))
            print(f"Current epoch training loss: {self.epoch_losses[-1]}")

            with torch.no_grad():
                self.model.eval()
                dev_avg_loss = 0
                for inputs, labels in tqdm(dev_dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    in_dist_indices = torch.isin(labels, self.anomaly_labels, invert=True)
                    out_dist_indices = torch.isin(labels, self.anomaly_labels)

                    outputs = self.model(inputs)

                    in_dist_outputs = outputs[[in_dist_indices.squeeze()]]
                    in_dist_labels = labels[[in_dist_indices.squeeze()]]

                    out_dist_outputs = outputs[[out_dist_indices.squeeze()]]
                    ce_loss = self.ce_loss(in_dist_outputs, in_dist_labels)

                    # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
                    oe_loss = self.lambda_hp * -(
                        out_dist_outputs.mean(1) - torch.logsumexp(out_dist_outputs, dim=1)
                    ).mean()

                    loss = ce_loss + oe_loss
                    dev_avg_loss += float(loss)

                print(f"Validation loss: {dev_avg_loss / len(dev_dataloader)}")


    def test(self, test_dataloader):
        pass

    def eval(self, eval_dataloader):
        pass

    def get_accuracy(self):
        pass


    def calc_oe_loss(self, model_output):
        pass


def train():
    pass


def test():
    pass


def run_train_with_oe():
    pass


def main():
    torch.random.manual_seed(42)

    batch_size = 128

    cifar10_dataset = get_cifar10_dataset()

    train_idx, test_idx, dev_idx = torch.utils.data.random_split(
        cifar10_dataset,
        [0.8, 0.1, 0.1]
    )

    anomaly_labels = [0, 5]
    invalid = torch.tensor(anomaly_labels)

    # a = torch.randint(10, (10, 1))
    # print(a)
    # in_dist_indices = torch.isin(a, invalid)
    # print(in_dist_indices.float())
    # print(a[[in_dist_indices.squeeze()]])

    # out_dist_indices = torch.isin(a, invalid)
    # print(a[[out_dist_indices.squeeze()]])
    # exit()

    train_dl = torch.utils.data.DataLoader(
        train_idx, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_idx, batch_size=batch_size, shuffle=True)
    dev_dl = torch.utils.data.DataLoader(
        dev_idx, batch_size=batch_size, shuffle=True)

    wide_resnet_oe = WideResnetOutlierExposure(
        anomaly_labels,
        epochs=10,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available else "cpu")
    wide_resnet_oe.train(train_dl, dev_dl)

    import matplotlib.pyplot as plt

    plt.plot(wide_resnet_oe.epoch_accuracy)
    plt.show()


if __name__ == "__main__":
    main()
