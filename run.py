import copy
import numpy as np
import torch
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Dataset
from load_data import TrainDS,TestDS
from model import Siamese
from split_dataset import split_train_test
from metrics import AA_andEachClassAccuracy,Kappa
from loss_function import PromptLoss,ContrastiveLoss
def train(model,epochs, classiSet, classiLoader, loss_fn_prompt, loss_fn_classi, loss_fn_contrastive, device, optimizer, scheduler):
    best_model = copy.deepcopy(model)
    best_loss=100000
    for epoch in range(1, epochs + 1):
        classi_loss_total = 0
        loss_prompt_total = 0
        loss_contrastive_total = 0
        train_acc = 0
        for img1, img2, label1,label2,label in classiLoader:
            model.train()
            img1, img2, label1,label2,label = img1.to(device), img2.to(device), label1.to(device),label2.to(device),label.to(device)
            z_1,center_1,prompt_outputs1 = model(img1)
            z_2,center_2,prompt_outputs2 = model(img2)
            
            # 计算三个独立的提示损失
            loss_prompt1 = loss_fn_prompt(prompt_outputs1[0],prompt_outputs2[0], label)
            # loss_prompt2 = loss_fn_prompt(prompt_outputs1[1],prompt_outputs2[1], label)
            loss_prompt = loss_prompt1
            loss_prompt_total += loss_prompt.cpu().item()

            # 计算对比损失
            loss_contrastive = loss_fn_contrastive(center_1, center_2, label)
            loss_contrastive_total += loss_contrastive.cpu().item()
            
            # 计算分类损失
            classi_loss1 = loss_fn_classi(z_1 , label1)
            classi_loss2 = loss_fn_classi(z_2 , label2)
            classi_loss_total += classi_loss1.cpu().item() + classi_loss2.cpu().item()

            pred1 = torch.max(z_1, 1)[1]
            pred2 = torch.max(z_2, 1)[1]
            train_acc += (pred1 == label1).sum().cpu().item() + (pred2 == label2).sum().cpu().item()
            loss = classi_loss1 + classi_loss2 + loss_prompt + 0.1*loss_contrastive

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 在每个epoch结束后打印统计信息
        print(
            f'Epoch:{epoch}   OA:{train_acc / (len(classiSet)*2)*100:.2f}   '
            f'loss_classi:{classi_loss_total / len(classiLoader):.5f}   '
            f'loss_prompt:{loss_prompt_total / len(classiLoader):.5f}   '
            f'loss_contrastive:{loss_contrastive_total / len(classiLoader):.5f}   '
        )

        if classi_loss_total < best_loss:
            best_loss = classi_loss_total
            best_model = copy.deepcopy(model)
            
        # 在每个epoch结束时更新学习率
        scheduler.step()
    return best_model

def test(model, testLoader, device, output_units):
    test_acc = 0
    predictions = []
    confusion_matrix = np.zeros((output_units, output_units), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(testLoader, desc='Testing', unit='batch'):
            img, label = img.to(device), label.to(device)
            z, _, _ = model(img)
            pred = torch.max(z, 1)[1]
            test_acc += (pred == label).sum().cpu().item()
            predictions += z.argmax(dim=1).cpu().numpy().tolist()
            for confu in range(img.shape[0]):
                confusion_matrix[label[confu], pred[confu]] += 1
    OA = test_acc / len(predictions)
    each_acc, AA = AA_andEachClassAccuracy(confusion_matrix)
    kappa = Kappa(confusion_matrix)
    print(f"OA:{oa*100:.2f}")
    return OA, AA, kappa, each_acc, predictions,confusion_matrix

best_combinations = {
    'SA': {'pool_size': 80, 'top_k': 6},
    'bo': {'pool_size': 20, 'top_k': 2},
    'XuZhou': {'pool_size': 80, 'top_k': 4},
    'HU2013': {'pool_size': 100, 'top_k': 2},
    'HanChuan': {'pool_size': 20, 'top_k': 2},
    'HongHu': {'pool_size': 60, 'top_k': 2}
}

# 定义要测试的数据集和perclass值
dataset = 'HanChuan'
seed = 345
perclass_value = 5
K = 32 
windowSize = 9 
length = 32 
batch_size = 128    
epochs = 150
lr = 1e-2
output_units = 16
pool_size = best_combinations[dataset]['pool_size']
top_k = best_combinations[dataset]['top_k']
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
loss_fn_classi = torch.nn.CrossEntropyLoss().to(device)
loss_fn_prompt = PromptLoss().to(device)
loss_fn_contrastive = ContrastiveLoss().to(device)

print(f"\n================ 开始处理数据集: {dataset} ================")
print(f"使用最佳参数组合: pool_size={pool_size}, top_k={top_k}")
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"\n================ 训练种子: {seed} ================\n")
model = Siamese(deep=K,
                output_units=output_units,
                windowSize=windowSize,
                pool_size=pool_size,
                length=length,
                top_k=top_k).to(device)
Xtrain, Xtest, ytrain, ytest, _,_,pos_train, pos_test = split_train_test(dataset=dataset, K=K, seed=seed, perclass=perclass, windowSize=windowSize)
optim_classi = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_classi, T_max=epochs, eta_min=0.01*lr)
classiSet = TrainDS(Xtrain, ytrain)
classiLoader = DataLoader(classiSet, batch_size=batch_size, shuffle=True, drop_last=False)
testSet = TestDS(Xtest, ytest)
testLoader = DataLoader(testSet, batch_size=20480, shuffle=False, drop_last=False)

#训练模型
train_start = time.time()
model = train(
    model=model,
    epochs=epochs,
    classiSet=classiSet,
    classiLoader=classiLoader,
    loss_fn_prompt=loss_fn_prompt,
    loss_fn_classi=loss_fn_classi,
    loss_fn_contrastive=loss_fn_contrastive,
    device=device,
    optimizer=optim_classi,
    scheduler=scheduler
)
train_end = time.time()
train_time = train_end - train_start

#测试模型
test_start = time.time()
oa, aa, kappa, each_acc, predictions,confusion_matrix = test(
    model=model,
    testLoader=testLoader,
    device=device,
    output_units=output_units
)
test_end = time.time()
test_time = test_end - test_start

print(f'seed:{seed}  OA:{oa*100:.2f}  训练时间:{train_time:.2f}s  测试时间:{test_time:.2f}s')
