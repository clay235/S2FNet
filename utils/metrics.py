import numpy as np


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(list_diag / list_raw_sum)
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def Kappa(confusion_matrix):
    observed_agreement = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    total_samples = np.sum(confusion_matrix)
    expected_agreement = np.sum((row_sums * col_sums) / total_samples) / total_samples
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa
