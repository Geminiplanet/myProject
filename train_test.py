import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def train_epoch(model, criterion, optimizer, dataloaders):
    model.train()
    losses = []
    y_label_list = {}
    y_pred_list = {}
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        batch_x, batch_l, batch_p = data
        optimizer.zero_grad()
        pred = model(batch_x, batch_l)
        loss = 0
        for i in range(12):
            y_pred = pred[:, i * 2:(i + 1) * 2]
            y_label = batch_p[:, i].squeeze()
            validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
            if len(validId) == 0:
                continue
            y_pred = y_pred[validId]
            y_label = y_label[validId]
            loss += criterion[i](y_pred, y_label)
            y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(12)]
    train_prc = [metrics.auc(precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                             precision_recall_curve(y_label_list[i], y_pred_list[i])[0]) for i in range(12)]
    train_loss = np.array(losses).mean()
    return train_loss, np.array(train_roc).mean(), np.array(train_prc).mean()


def train(model, criterion, optimizer, scheduler, dataloaders, epochs):
    print('>> Train a Model')
    best_acc = 0.

    for epoch in range(epochs):
        train_loss, train_roc, train_prc = train_epoch(model, criterion, optimizer, dataloaders)
        scheduler.step()
        print(f'epoch {epoch}: train loss is {train_loss}, train_roc: {train_roc}, train_prc: {train_prc}')
        # if epoch % 20 == 19:
        #     roc_auc, prc_auc = test(model, dataloaders, mode='val')
        #     if best_acc < roc_auc:
        #         best_acc = roc_auc
        #         print(f'Val acc: {roc_auc}')
    print('>> Finished.')


def test(model, dataloaders, mode='test'):
    assert mode == 'val' or mode == 'test'
    model.eval()
    losses = []

    total = 0
    correct = 0
    with torch.no_grad():
        y_label_list = {}
        y_pred_list = {}
        for data in tqdm(dataloaders[mode], leave=False, total=len(dataloaders[mode])):
            batch_x, batch_l, batch_p = data
            pred = model(batch_x, batch_l)
            for i in range(12):
                y_pred = pred[:, i * 2:(i + 1) * 2]
                y_label = batch_p[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                y_pred = y_pred[validId]
                y_label = y_label[validId]
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                try:
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
        roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(12)]
        prc = [metrics.auc(precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                                 precision_recall_curve(y_label_list[i], y_pred_list[i])[0]) for i in range(12)]

    return np.array(roc).mean(), np.array(prc).mean()
