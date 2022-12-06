import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import itertools
import train
from ensemble_ad.cifar10_data.dataset_cifar import load_CIFAR10_anomaly_data
from torchvision.transforms.functional import rotate, affine

parser = argparse.ArgumentParser(description='GOAD implementation')
parser.add_argument('--config', default='./config.yaml')

def transform_data(data):
    flips = (True, False)
    trans_x = (0, -8, 8)
    trans_y = (0, -8, 8)
    rotations = range(4)
    n_trans = len(flips)*len(trans_x)*len(trans_y)*len(rotations)
    trans_inds = np.tile(np.arange(n_trans), len(data))
    trans_data = np.repeat(np.array(data), n_trans, axis=0)
    for i, (idx, flip, tx, ty, k_rotation) in enumerate(itertools.product(range(data.shape[0]),flips,trans_x,trans_y,rotations)):
        if flip:
            trans_data[i] = np.flip(trans_data[i], 1)
        if tx != 0 or ty != 0:
            trans_data[i] = np.roll(trans_data[i], shift=(ty,tx), axis=(0,1))
        if k_rotation != 0:
            trans_data[i] = np.rot90(trans_data[i], k=k_rotation, axes=(0,1))
    return trans_data, trans_inds


def load_trans_data(args):
    x_train, x_test, y_test = load_CIFAR10_anomaly_data(label=args.class_ind)
    x_train_trans, trans_labels = transform_data(x_train)
    x_test_trans, _ = transform_data(x_test)
    x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
    y_test = np.array(y_test) == args.class_ind
    return x_train_trans, trans_labels, x_test_trans, y_test


def train_anomaly_detector(args):
    x_train_trans, trans_labels, x_test_trans, y_test = load_trans_data(args)
    trClf = train.TransClassifier(np.unique(trans_labels).size, args)
    trClf.fit_trans_classifier(x_train_trans, trans_labels, x_test_trans, y_test)


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    train_anomaly_detector(args)


if __name__ == '__main__':
    main()