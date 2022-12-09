import numpy as np
import torch as torch
import itertools


flips = (True, False)
trans_x = (0, -8, 8)
trans_y = (0, -8, 8)
rotations = range(4)


def num_trans():
    return len(flips)*len(trans_x)*len(trans_y)*len(rotations)


def transform_data(data):
    n_trans = num_trans()
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