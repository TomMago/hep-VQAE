import numpy as np
import matplotlib.pyplot as plt

def test_collapse(y_true, y_recon):


    #y_true_shuffle = np.copy(y_true)
    #np.random.shuffle(y_true_shuffle)
    #
    #y_recon_shuffle = np.copy(y_recon)
    #np.random.shuffle(y_recon_shuffle)

    p = np.random.permutation(y_true.shape[0])
    y_true_shuffle = y_true[p]
    y_recon_shuffle = y_recon[p]

    true_diff = np.sum(np.abs(y_true - y_true_shuffle))
    recon_diff = np.sum(np.abs(y_recon - y_recon_shuffle))
    return recon_diff / true_diff


def intensity_hist(y_true, y_recon):
    y_true = y_true.flatten()
    y_recon = y_recon.flatten()

    idx = np.argsort(y_true)

    relatives = y_recon[idx] / y_true[idx]
    return relatives
