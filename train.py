import numpy as np
import torch
from torch.utils.data import Dataset


class UserDataset(Dataset):
    def __init__(self, data_dir):
        self.adj_data = torch.tensor(np.load(data_dir + 'adj_data.npy'), dtype=torch.long)
        self.e_data = torch.tensor(np.load(data_dir + 'e_data.npy'), dtype=torch.long)
        self.f_data = torch.tensor(np.load(data_dir + 'f_data.npy'), dtype=torch.long)
        self.y_data = torch.tensor(np.load(data_dir + 'y_data.npy'), dtype=torch.long)
        self.len = self.adj_data.shape[0]

    def __getitem__(self, index):
        return {
            'adj': self.adj_data[index], 'e_feat': self.e_data[index], 'a_feat': self.f_data[index],
            'y': self.y_data[index]
        }

    def __len__(self):
        return self.len



