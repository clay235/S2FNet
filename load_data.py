import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import random

class TrainDS(Dataset):
    def __init__(self, Xtrain, ytrain,data_augmentation=True):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        if np.random.random() > 0.5:
            data1=self.x_data[index]
            if self.data_augmentation:
                data1=self.augment_data(data1)
            label1=self.y_data[index]

            k=random.randint(0,self.len-1)
            while k==index:
                k=random.randint(0,self.len-1)
            
            data2=self.x_data[k]
            if self.data_augmentation:
                data2=self.augment_data(data2)
            label2=self.y_data[k]

            if self.y_data[index]==self.y_data[k]:
                label=torch.tensor(1, dtype=torch.long)  
            else:
                label=torch.tensor(0, dtype=torch.long)
        else:
            data1=self.x_data[index]
            if self.data_augmentation:
                data1=self.augment_data(data1)
            label1=self.y_data[index]

            data2=self.x_data[index]
            if self.data_augmentation:
                data2=self.augment_data(data2)
            label2=self.y_data[index]

            label=torch.tensor(1, dtype=torch.long)
        return data1, data2, label1,label2,label

    def __len__(self):
        return self.len
    
    @staticmethod
    def augment_data(data):
        if np.random.random() > 0.5:
            # 随机选择一种数据增强方法
            if np.random.random() > 0.5:
                prob = np.random.random()
                if 0 <= prob <= 0.2:
                    data = torch.flip(data, dims=[-1])  # 水平翻转
                elif 0.2 < prob <= 0.4:
                    data = torch.flip(data, dims=[-2])  # 垂直翻转
                elif 0.4 < prob <= 0.6:
                    data = torch.rot90(data, k=1, dims=[-2, -1])  # 旋转90度
                elif 0.6 < prob <= 0.8:
                    data = torch.rot90(data, k=2, dims=[-2, -1])  # 旋转180度
                elif 0.8 < prob <= 1.0:
                    data = torch.rot90(data, k=3, dims=[-2, -1])  # 旋转270度
        return data

class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        data = self.x_data[index]
        label = self.y_data[index]
        return data,label
    def __len__(self):
        return self.len