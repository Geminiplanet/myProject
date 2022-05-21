import argparse
import random

import torch.cuda
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from config import *
from dataset import load_tox21_data
from model import Predictor
from selection_methods import query_samples
from train_test import train, test

parser = argparse.ArgumentParser()
args = parser.parse_args()


def main():
    # load training and testing dataset
    train_data, test_data = load_tox21_data('data/tox21.csv')
    print(f'train_dataset: {len(train_data)}, test_dataset: {len(test_data)}')
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    labeled_set = indices[:ADDENDUM]
    unlabeled_set = [x for x in indices if x not in labeled_set]
    train_loader = DataLoader(train_data, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=BATCH)
    dataloaders = {'train': train_loader, 'test': test_loader}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for cycle in range(CYCLES):
        # Model - create new instance for every cycle so that it resets
        para = {'Nseq': 201, 'Nfea': len(CHAR_LIST), 'hidden_dim': 512,
                'seed_dim': 512, 'NLSTM_layer': 1, 'device': device}
        predictor = Predictor(para)
        # models = {'backbone': predictor}

        # Loss, criterion and scheduler (re)initialization
        criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(device), reduction='mean') for w in
                     train_data.weights]
        optimizer = optim.Adam(predictor.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
        # optimizer_cri = optim.Adam(net.Cri.parameters(), lr=0.000002)

        # training and testing
        train(predictor, criterion, optimizer, scheduler, dataloaders, EPOCHP)
        roc_auc, prc_auc = test(predictor, dataloaders, mode='test')
        print(f'Cycle {cycle + 1}/{CYCLES} || labeled data size {len(labeled_set)}, test acc = {roc_auc}')
        if cycle == CYCLES - 1:
            print('Finished.')
            break
        # Get the indices of the unlabeled samples to train on next cycle
        arg = query_samples(models, 'TA-VAAL', train_data, unlabeled_set, labeled_set, cycle, args)

        # Update the labeled dataset and the unlabeled dataset, respectively
        new_list = list(torch.tensor(unlabeled_set)[arg][:ADDENDUM].numpy())
        # print(len(new_list), min(new_list), max(new_list))
        labeled_set += list(torch.tensor(unlabeled_set)[arg][-ADDENDUM:].numpy())
        listd = list(torch.tensor(unlabeled_set)[arg][:-ADDENDUM].numpy())
        unlabeled_set = listd + unlabeled_set[SUBSET:]
        print(len(labeled_set), min(labeled_set), max(labeled_set))
        # Create a new dataloader for the updated labeled dataset
        dataloaders['train'] = DataLoader(train_data, batch_size=BATCH,
                                          sampler=SubsetRandomSampler(labeled_set),
                                          pin_memory=True)


if __name__ == '__main__':
    main()
