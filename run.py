import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mydataset import TrainDS, TestDS
from model import S2FNet
from sklearn.metrics import confusion_matrix
from split_dataset import split_train_test
from tqdm import tqdm
import time
import os
import json
import argparse
from loss_function import PromptLoss, ContrastiveLoss
from utils.metrics import AA_andEachClassAccuracy, Kappa
from params import print_model_stats


def train(model, epochs, classiSet, classiLoader, loss_fn_prompt, loss_fn_classi, loss_fn_contrastive, device, optimizer, scheduler):
    best_model = copy.deepcopy(model)
    best_loss = 100000
    for epoch in range(1, epochs + 1):
        classi_loss_total = 0
        loss_prompt_total = 0
        loss_contrastive_total = 0
        train_acc = 0
        for img1, img2, label1, label2, label in classiLoader:
            model.train()
            img1, img2, label1, label2, label = img1.to(device), img2.to(device), label1.to(device), label2.to(device), label.to(device)
            z_1, center_1, prompt_outputs1 = model(img1)
            z_2, center_2, prompt_outputs2 = model(img2)

            loss_prompt1 = loss_fn_prompt(prompt_outputs1[0], prompt_outputs2[0], label)
            loss_prompt = loss_prompt1
            loss_prompt_total += loss_prompt.cpu().item()

            loss_contrastive = loss_fn_contrastive(center_1, center_2, label)
            loss_contrastive_total += loss_contrastive.cpu().item()

            classi_loss1 = loss_fn_classi(z_1, label1)
            classi_loss2 = loss_fn_classi(z_2, label2)
            classi_loss_total += classi_loss1.cpu().item() + classi_loss2.cpu().item()

            pred1 = torch.max(z_1, 1)[1]
            pred2 = torch.max(z_2, 1)[1]
            train_acc += (pred1 == label1).sum().cpu().item() + (pred2 == label2).sum().cpu().item()
            loss = classi_loss1 + classi_loss2 + loss_prompt + 0.1 * loss_contrastive

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f'Epoch:{epoch}   OA:{train_acc / (len(classiSet) * 2) * 100:.2f}   '
            f'loss_classi:{classi_loss_total / len(classiLoader):.4f}   '
        )

        if classi_loss_total < best_loss:
            best_loss = classi_loss_total
            best_model = copy.deepcopy(model)

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

    return OA, AA, kappa, each_acc, predictions, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='S2FNet Training')
    parser.add_argument('--dataset', type=str, default='HanChuan', help='Dataset name (options: SA, bo, XuZhou, HU2013, HanChuan, HongHu)')
    parser.add_argument('--seed', type=int, default=345, help='Random seed')
    parser.add_argument('--perclass', type=int, default=5, help='Samples per class')
    parser.add_argument('--K', type=int, default=32, help='PCA components')
    parser.add_argument('--windowSize', type=int, default=9, help='Window size')
    parser.add_argument('--length', type=int, default=32, help='Prompt length')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:7', help='Device to use')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = args.seed
    dataset = args.dataset
    perclass = args.perclass
    K = args.K
    windowSize = args.windowSize
    length = args.length
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loss_fn_classi = torch.nn.CrossEntropyLoss().to(device)
    loss_fn_prompt = PromptLoss().to(device)
    loss_fn_contrastive = ContrastiveLoss().to(device)

    pool_size = config[dataset]['pool_size']
    top_k = config[dataset]['top_k']
    output_units = config[dataset]['output_units']

    print(f"\n================ Start training: {dataset} ================")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = S2FNet(deep=K,
                    output_units=output_units,
                    windowSize=windowSize,
                    pool_size=pool_size,
                    length=length,
                    top_k=top_k).to(device)
    
    print_model_stats(model, dataset, K, length, windowSize, pool_size, top_k, output_units)
    
    Xtrain, Xtest, ytrain, ytest, _, _, pos_train, pos_test = split_train_test(dataset=dataset, K=K, seed=seed, perclass=perclass, windowSize=windowSize)
    optim_classi = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_classi, T_max=epochs, eta_min=0.01 * lr)

    classiSet = TrainDS(Xtrain, ytrain)
    classiLoader = DataLoader(classiSet, batch_size=batch_size, shuffle=True, drop_last=False)
    testSet = TestDS(Xtest, ytest)
    testLoader = DataLoader(testSet, batch_size=20480, shuffle=False, drop_last=False)

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

    oa, aa, kappa, each_acc, predictions, confusion_matrix = test(
        model=model,
        testLoader=testLoader,
        device=device,
        output_units=output_units
    )

    print(f'seed: {seed}  OA: {oa*100:.2f}  AA: {aa*100:.2f}  kappa: {kappa*100:.2f}')

    os.makedirs(f'{dataset}/{perclass}/pth', exist_ok=True)
    os.makedirs(f'{dataset}/{perclass}/confusion', exist_ok=True)
    torch.save(model.state_dict(), f'{dataset}/{perclass}/pth/{oa*100:.2f}.pth')
    np.save(f'{dataset}/{perclass}/confusion/{oa*100:.2f}.npy', confusion_matrix)


if __name__ == '__main__':
    main()
