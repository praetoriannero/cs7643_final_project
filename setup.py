import os
from setuptools import setup, find_packages
import subprocess as sp

print("Beginning setup")
setup(
	name="ensemble_ad",
	version="0.0.1",
	author="Anomaly Detectives",
	packages=find_packages()
	)

print("Downloading CIFAR-10 dataset...")
current_dir = os.getcwd()
os.chdir(os.path.join(current_dir, "ensemble_ad", "cifar10-data"))
downloaded_cifar_data = False
if os.name == "nt":
	sp.run(["get_data.bat"])
	downloaded_cifar_data = True
else:
	sp.run([r"get_data.sh"])
	downloaded_cifar_data = True

os.chdir(current_dir)

print("Downloading SVHN dataset...")
os.chdir(os.path.join(current_dir, "ensemble_ad", "svhn-data"))
downloaded_svhn_data = False
if os.name == "nt":
	sp.run(["get_data.bat"])
	downloaded_svhn_data = True
else:
	sp.run([r"get_data.sh"])
	downloaded_svhn_data = True

os.chdir(current_dir)

if downloaded_svhn_data and downloaded_cifar_data:
	print("Project setup completed successfully!")
else:
	print("Error! Could not complete download of CIFAR-10 dataset.")
