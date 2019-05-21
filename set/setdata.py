import numpy as np
import os
import pickle
from torch.utils import data

class SetDataset(data.Dataset):
    def __init__(self, data_dir, dataset_type='train'):
        with open(data_dir, 'rb') as file:
            synthetic_data = pickle.load(file, encoding='latin-1')
        self._synthetic_data = np.array(synthetic_data[dataset_type], np.float32)
        self._n = len(self._synthetic_data)

    def __getitem__(self, item):
        return self._synthetic_data[item]

    def __len__(self):
        return self._n

