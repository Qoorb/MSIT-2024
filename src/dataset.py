import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from scipy.io import loadmat


class EEGDataset(Dataset):
    def __init__(self, save_path='./data', dir_files='',
                 transform=None, group_size=1000, Normalize=True) -> None:
        self.save_path = save_path
        self.dir_files = dir_files
        self.transform = transform
        self.group_size = group_size
        self.normalize = Normalize
        self.data = self._read_data(Normalize)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        data, label = self.data[idx]

        if self.transform:
            data = self.transform(data)
            
        return data, label
    
    # inner function
    def _info(self) -> tuple:
        return len(self.data), self.data[0].shape
    
    def _normalize(self, values, epsilon=1e-7) -> list:
        values = torch.from_numpy(np.asarray(values))

        mean = torch.mean(values)
        std = torch.std(values)

        return ((values - mean) / (std + epsilon)).detach().cpu().numpy()

    def _read_data(self, Normalize) -> list:
        files = [file for file in os.listdir(self.dir_files) if file.endswith('.mat')]
        dataset = []
        labels = []

        for file in files[:2]:
            data = loadmat(os.path.join(self.dir_files, file))['EEG'][0][0][15][:64] # 64 - EEG channels
            label = 0 if file.split('_') in CTLSX else 1

            for start_idx in range(0, data.shape[1], self.group_size):
                if start_idx + self.group_size <= data.shape[1]:
                    dataset.append(data[:, start_idx:start_idx + self.group_size])
                    labels.append(label)

            print(f"{file} added")
        
        if Normalize:
            dataset = self._normalize(dataset)

        return list(zip(dataset, labels))
    
    def save(self) -> None:
        pickle.dump(self, open(f"{self.save_path}/train_dataset_EEG.pkl", 'wb'), 4) # TODO: fix

        print('dataset has been saved')

    @staticmethod
    def load_dataset(file_name: str) -> object:
        data = pickle.load(open(file_name,'rb'))
        
        print('dataset has been loaded')
        return data


if __name__ == '__main__':

    # CONST
    CTLSX = [894, 908, 8010, 890, 891,
             892, 893, 895, 896, 897,
             898, 899, 900, 901, 902,
             903, 904, 905, 906, 907,
             909, 910, 911, 912, 913,
             914, 8060, 8070] # class - 0 (Control group)

    transform = transforms.Compose([
        transforms.ToTensor() # transforms.Normalize(-1, 1)
    ])

    data = EEGDataset(dir_files='./data/d002', transform=transform, Normalize=True)
    data.save()
