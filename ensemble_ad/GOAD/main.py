import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import itertools
from ensemble_ad.cifar10_data.dataset_cifar import load_CIFAR10_anomaly_data
import train
from torchvision.transforms.functional import affine

def transform_data(data):
    flips = (True, False)
    trans_x = (0, -8, 8)
    trans_y = (0, -8, 8)
    rotations = range(4)
    n_trans = len(flips)*len(trans_x)*len(trans_y)*len(rotations)
    trans_inds = np.tile(np.arange(n_trans), len(data))
    trans_data = torch.from_numpy(np.repeat(np.array(data), n_trans, axis=0))
    trans_data = trans_data.permute((0,3,1,2))
    for i, (idx, flip, tx, ty, k_rotation) in enumerate(itertools.product(range(data.shape[0]),flips,trans_x,trans_y,rotations)):
        if flip:
            trans_data[i] = torch.fliplr(trans_data[i])
        if tx != 0 or ty != 0:
            trans_data[i] = affine(img=trans_data[i], angle=0, translate=[tx,ty], scale=1, shear=0)
        if k_rotation != 0:
            trans_data[i] = affine(img=trans_data[i], angle=k_rotation*90, translate=[0,0], scale=1, shear=0)
    # trans_data = trans_data.permute((0,2,3,1))
    return trans_data, trans_inds

def load_trans_data():
    x_train, x_test, y_test = load_CIFAR10_anomaly_data(label=1)
    x_train_trans, labels = transform_data(x_train)
    x_test_trans, _ = transform_data(x_test)
    # x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
    y_test = torch.from_numpy(np.array(y_test)) == 1 #args.class_ind
    return x_train_trans, x_test_trans, y_test

def train_anomaly_detector():
    pass

x_train, x_test, y_test = load_trans_data()

for i in range(20):
    im = x_train[i].permute((1,2,0)).numpy()
    plt.imshow(im)
    plt.show()