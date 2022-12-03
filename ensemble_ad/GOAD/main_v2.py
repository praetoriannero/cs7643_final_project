import argparse
import transformations as ts
import numpy as np
import matplotlib.pyplot as plt
from ensemble_ad.cifar10_data.dataset_cifar import load_CIFAR10_anomaly_data

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def load_trans_data(trans):
    x_train, x_test, y_test = load_CIFAR10_anomaly_data(label=1)
    x_train_trans, labels = transform_data(x_train, trans)
    x_test_trans, _ = transform_data(x_test, trans)
    x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
    y_test = np.array(y_test) == 1
    return x_train_trans, x_test_trans, y_test

def train_anomaly_detector():
    pass

transformer = ts.get_transformer('complicated')
x_train, x_test, y_test = load_trans_data(transformer)

for i in range(20):
    im = x_train[i].transpose((1,2,0))
    plt.imshow(im)
    plt.show()