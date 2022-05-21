import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import *


def process_tox21_smiles_data(data_dir):
    char_dict = dict()
    i = 0
    for c in CHAR_LIST:
        char_dict[c] = i
        i += 1
    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in CHAR_LIST:
        if len(key) == 1:
            char_list1 += [key]
            char_dict1[key] = char_dict[key]
        elif len(key) == 2:
            char_list2 += [key]
            char_dict2[key] = char_dict[key]
        else:
            print("strange ", key)
    df = pd.read_csv(data_dir)
    target = df[TOX21_TASKS].values
    smiles_list = df.smiles.values
    Xdata = []
    Ldata = []
    Pdata = []
    for i, smi in enumerate(smiles_list):
        # print(type(line[0]))
        # line[0]: NR-AR, line[13]: smiles
        smiles_len = len(smi)
        if smiles_len > 200:
            continue
        label = target[i]
        label[np.isnan(label)] = 6
        Pdata.append(label)
        X_d = np.zeros([MAX_SEQ_LEN + 1], dtype=int)
        X_d[0] = char_dict['<']
        j = 0
        istring = 0
        check = True
        while check:
            char2 = smi[j: j + 2]
            char1 = smi[j]
            if char2 in char_list2:
                index = char_dict2[char2]
                j += 2
                if j >= smiles_len:
                    check = False
            elif char1 in char_list1:
                index = char_dict1[char1]
                j += 1
                if j >= smiles_len:
                    check = False
            else:
                print(char1, char2, "error")
                sys.exit()
            X_d[istring + 1] = index
            istring += 1
        for k in range(istring, MAX_SEQ_LEN - 1):
            X_d[k + 1] = char_dict['>']
        Xdata.append(X_d)
        Ldata.append(istring + 1)
    weights = []
    for i, task in enumerate(TOX21_TASKS):
        negative_df = df[df[task] == 0][["smiles", task]]
        positive_df = df[df[task] == 1][["smiles", task]]
        neg_len = len(negative_df)
        pos_len = len(positive_df)
        weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    train_num = int(0.7 * len(Xdata))
    X_train = np.asarray(Xdata[:train_num], dtype="long")
    L_train = np.asarray(Ldata[:train_num], dtype="long")
    P_train = np.asarray(Pdata[:train_num], dtype="long")
    X_test = np.asarray(Xdata[train_num:], dtype="long")
    L_test = np.asarray(Ldata[train_num:], dtype="long")
    P_test = np.asarray(Pdata[train_num:], dtype="long")
    Weights = np.asarray(weights, dtype="float")
    print(X_train.shape, L_train.shape, P_train.shape, Weights.shape)
    np.save('data/X_train.npy', X_train)
    np.save('data/L_train.npy', L_train)
    np.save('data/P_train.npy', P_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/L_test.npy', L_test)
    np.save('data/P_test.npy', P_test)
    np.save('data/Weights.npy', Weights)
    return


class UserDataset(Dataset):
    def __init__(self, data_dir, name):
        self.Xdata = torch.tensor(np.load(data_dir + f'X_{name}.npy'), dtype=torch.long)
        self.Ldata = torch.tensor(np.load(data_dir + f'L_{name}.npy'), dtype=torch.long)
        self.Pdata = torch.tensor(np.load(data_dir + f'P_{name}.npy'), dtype=torch.long)
        self.len = self.Xdata.shape[0]
        if name == 'train':
            self.weights = torch.tensor(np.load(data_dir + 'Weights.npy'), dtype=torch.float)

    def __getitem__(self, index):
        return self.Xdata[index], self.Ldata[index], self.Pdata[index]

    def __len__(self):
        return self.len


def load_tox21_data(data_dir):
    # process_tox21_smiles_data(data_dir)
    train_data = UserDataset('data/', 'train')
    test_data = UserDataset('data/', 'test')
    return train_data, test_data


def main():
    load_tox21_data('data/tox21.csv')


if __name__ == "__main__":
    main()
