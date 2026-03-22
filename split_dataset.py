import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import scipy.io as sio
import cv2 as cv

def split_train_test(dataset, K, seed=345, perclass=5, windowSize=9):
    def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX, pca

    def loadData(name):
        possible_paths = [
            '/data02/zhangqinhan/pc/datasets',
            '/data02/zhangqinhan/dataset',
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break

        if data_path is None:
            raise FileNotFoundError('找不到有效的数据路径，请确保至少有一个路径存在')

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
        elif name == 'co':
            data = sio.loadmat('/data02/zhangqinhan/datasets/CopratesChasma.mat')['CopratesChasma']
            labels = sio.loadmat('/data02/zhangqinhan/datasets/CopratesChasma_train_gt.mat')['CopratesChasma_gt']
        elif name == 'me':
            data = sio.loadmat('/data02/zhangqinhan/datasets/MelasChasma.mat')['MelasChasma']
            labels = sio.loadmat('/data02/zhangqinhan/datasets/MelasChasma_train_gt.mat')['MelasChasma_gt']
        elif name == 'ga':
            data = sio.loadmat('/data02/zhangqinhan/datasets/GaleCrater.mat')['GaleCrater']
            labels = sio.loadmat('/data02/zhangqinhan/datasets/GaleCrater_train_gt.mat')['GaleCrater_gt']
        elif name == 'nili':
            data = sio.loadmat('/data02/zhangqinhan/datasets/NiliFossae.mat')['NiliFossae']
            labels = sio.loadmat('/data02/zhangqinhan/datasets/NiliFossae_gt.mat')['NiliFossae_gt']
        elif name == 'holden':
            data = sio.loadmat('/data02/zhangqinhan/datasets/holden.mat')['holden']
            labels = sio.loadmat('/data02/zhangqinhan/datasets/holden_gt.mat')['holden_gt']
        elif name == 'utopia':
            data = sio.loadmat('/data02/zhangqinhan/datasets/Utopia.mat')['Utopia']
            labels = sio.loadmat('/data02/zhangqinhan/datasets/Utopia_gt.mat')['Utopia_gt']
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

    def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = padWithZeros(X, margin=margin)
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchesPositions = np.zeros((X.shape[0] * X.shape[1], 2), dtype=int)
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchesPositions[patchIndex, :] = [r - margin, c - margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            valid_indices = patchesLabels > 0
            patchesData = patchesData[valid_indices, :, :, :]
            patchesLabels = patchesLabels[valid_indices]
            patchesPositions = patchesPositions[valid_indices]
            patchesLabels -= 1

        return patchesData, patchesLabels, patchesPositions

    def split_train_test_data(X_all, y_all, positions_all, perclass, seed):
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

        Xtrain = X_all[Xtrain_indices]
        Xtest = X_all[Xtest_indices]
        ytrain = y_all[Xtrain_indices]
        ytest = y_all[Xtest_indices]
        Xtrain_positions = positions_all[Xtrain_indices]
        Xtest_positions = positions_all[Xtest_indices]

        return Xtrain, Xtest, ytrain, ytest, Xtrain_positions, Xtest_positions

    possible_paths = [
        '/data02/zhangqinhan/pc',
        '/data02/zhangqinhan/pc',
        'datasets',
    ]

    base_path = None
    for path in possible_paths:
        if os.path.exists(path):
            base_path = path
            break

    if base_path is None:
        raise FileNotFoundError('No valid dataset path found.')

    print(f'Loading {dataset} dataset...')
    X, y = loadData(dataset)
    output_units = y.max()

    if K != None:
        X, _ = applyPCA(X, numComponents=K)
    else:
        print('No PCA')

    for i in range(X.shape[2]):
        input_max = np.max(X[:, :, i])
        input_min = np.min(X[:, :, i])
        X[:, :, i] = (X[:, :, i] - input_min) / (input_max - input_min)

    X_all, y_all, positions_all = createImageCubes(X, y, windowSize=windowSize)
    Xtrain, Xtest, ytrain, ytest, Xtrain_positions, Xtest_positions = split_train_test_data(X_all, y_all, positions_all, perclass, seed)
    X_all = X_all.reshape(-1, windowSize, windowSize, K, 1)
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    X_all = X_all.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    return Xtrain, Xtest, ytrain, ytest, X_all, y_all, Xtrain_positions, Xtest_positions


if __name__ == '__main__':
    split_train_test('PU', 64, 345, 5, 3)
