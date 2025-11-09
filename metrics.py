import numpy as np
from operator import truediv
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def Kappa(confusion_matrix):
    # 计算观测一致性
    observed_agreement = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    # 计算每一行的和（真实标签的分布）
    row_sums = np.sum(confusion_matrix, axis=1)
    # 计算每一列的和（预测标签的分布）
    col_sums = np.sum(confusion_matrix, axis=0)
    # 计算总样本数
    total_samples = np.sum(confusion_matrix)
    # 计算期望一致性
    expected_agreement = np.sum((row_sums * col_sums) / total_samples) / total_samples
    # 计算Kappa系数
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa
