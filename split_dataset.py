import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import scipy.io as sio
from operator import truediv
import scipy.io as scio
import cv2 as cv

def split_train_test(dataset,K,seed=345,perclass=5,windowSize=9):
    def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX, pca


    def loadData(name):
        
        data_path = '/data02/zhangqinhan/pc/datasets'
        if name == 'IP':
            data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        elif name == 'SA':
            data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        elif name == 'PU':
            data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
            labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        elif name == 'KSC':
            data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
            labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        elif name == 'HU2018':
            data = sio.loadmat(os.path.join(data_path, 'HoustonU.mat'))['houstonU']
            labels = sio.loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
        elif name == 'HU2013':
            data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))['Houston']
            labels = sio.loadmat(os.path.join(data_path, 'Houston_gt.mat'))['Houston_gt']
        elif name == 'LongKou':
            data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
            labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        elif name == 'HanChuan':
            data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
            labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
        elif name == 'HongHu':
            data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
            labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
        elif name == 'XuZhou':
            data = sio.loadmat(os.path.join(data_path, 'xuzhou.mat'))['xuzhou']
            labels = sio.loadmat(os.path.join(data_path, 'xuzhou_gt.mat'))['xuzhou_gt']
        elif name == 'bo':
            data = sio.loadmat('/data02/zhangqinhan/pc/datasets/Botswana.mat')['Botswana']
            labels = sio.loadmat('/data02/zhangqinhan/pc/datasets/Botswana_gt.mat')['Botswana_gt']
        return data, labels


    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX


    def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = padWithZeros(X, margin=margin)
        # split patches
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchesPositions = np.zeros((X.shape[0] * X.shape[1], 2), dtype=int)  # 记录位置坐标
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchesPositions[patchIndex, :] = [r-margin, c-margin]  # 记录原始坐标
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            valid_indices = patchesLabels > 0
            patchesData = patchesData[valid_indices,:,:,:]
            patchesLabels = patchesLabels[valid_indices]
            patchesPositions = patchesPositions[valid_indices]  # 同时过滤位置坐标
            patchesLabels -= 1

        return patchesData, patchesLabels, patchesPositions


    def split_train_test(X_all, y_all, positions_all, perclass, seed):
        unique_classes = np.unique(y_all)
        Xtrain_indices = []
        Xtest_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(y_all == cls)[0]
            np.random.seed(seed)
            np.random.shuffle(cls_indices)
            train_indices = cls_indices[:perclass]
            test_indices = cls_indices[perclass:]
            Xtrain_indices.extend(train_indices)
            Xtest_indices.extend(test_indices)

        Xtrain_indices = np.array(Xtrain_indices)
        Xtest_indices = np.array(Xtest_indices)
        
        # 根据索引划分数据集
        Xtrain = X_all[Xtrain_indices]
        Xtest = X_all[Xtest_indices]
        ytrain = y_all[Xtrain_indices]
        ytest = y_all[Xtest_indices]
        Xtrain_positions = positions_all[Xtrain_indices]  # 训练集位置坐标
        Xtest_positions = positions_all[Xtest_indices]    # 测试集位置坐标
        
        return Xtrain, Xtest, ytrain, ytest, Xtrain_positions, Xtest_positions

    print(f'开始划分{dataset}')
    X, y = loadData(dataset)  # X为高光谱数据，y为标签
    if K != None:
        X, _ = applyPCA(X, numComponents=K)  # X为降维后的高光谱数据，K为降到的维度，不用管pca
    else:
        print('No PCA')

    for i in range(X.shape[2]):
        input_max = np.max(X[:, :, i])
        input_min = np.min(X[:, :, i])
        X[:, :, i] = (X[:, :, i] - input_min) / (input_max - input_min)

    X_all, y_all, positions_all = createImageCubes(X, y, windowSize=windowSize)
    Xtrain, Xtest, ytrain, ytest, Xtrain_positions, Xtest_positions = split_train_test(X_all, y_all, positions_all, perclass, seed)
    X_all = X_all.reshape(-1, windowSize, windowSize, K, 1)
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    X_all = X_all.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    return Xtrain, Xtest, ytrain, ytest, X_all, y_all, Xtrain_positions, Xtest_positions
if __name__ == '__main__':
    split_train_test('PU',64,345,5,3)