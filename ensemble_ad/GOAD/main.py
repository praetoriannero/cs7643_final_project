import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import itertools
import classifier as clf
import transforms as T
from ensemble_ad.cifar10_data.dataset_cifar import load_CIFAR10_anomaly_data


parser = argparse.ArgumentParser(description='GOAD implementation')
parser.add_argument('--config', default='./config.yaml')


def load_data(args):
    x_train, x_test, y_test = load_CIFAR10_anomaly_data(label=args.class_ind)
    return x_train, x_test, y_test


def train_GOAD(args):
    x_train, x_test, y_test = load_data(args)
    trClf = clf.GOADClassifier(T.num_trans(), args)
    trClf.train(x_train, x_test, y_test)


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    train_GOAD(args)


if __name__ == '__main__':
    main()