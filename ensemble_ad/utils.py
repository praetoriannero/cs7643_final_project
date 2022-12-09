from pathlib import Path
import torchvision.datasets as datasets
import torchvision as vision


def get_project_root_dir():
	# returns the system path of the "ensemble_ad" directory
	root_dir = Path(__file__).absolute().parent
	return root_dir


def get_svhn_dataset():
	# download the Street View House Number dataset and
	# store it in the "svhn-data" directory
	dataset = datasets.SVHN(
		root=get_project_root_dir() / "svhn-data",
		download=True,
		transform=vision.transforms.ToTensor())

	return dataset


def get_mnist_dataset():
	# download the Street View House Number dataset and
	# store it in the "svhn-data" directory
	dataset = datasets.MNIST(
		root=get_project_root_dir() / "mnist-data",
		download=True,
		transform=vision.transforms.ToTensor())

	return dataset


def get_cifar10_dataset():
	# download the CIFAR-10 dataset and store it in the "cifar10-data"
	# directory
	dataset = datasets.CIFAR10(
		root=get_project_root_dir() / "cifar10-data",
		download=True,
		transform=vision.transforms.ToTensor())

	return dataset


