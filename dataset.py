import os

import pickle

import torch
from torch.utils.data import Dataset

from scipy.io import loadmat


'''
TODO:
    The dataset class should contain data check?
    Add GPU processing?
    (63, 83501) != (63, 93501)?
'''


class EEGDataset(Dataset):
    def __init__(self, save_path='./data', dir_files='', transform=None) -> None:
        self.save_path = save_path
        self.dir_files = dir_files
        self.transform = transform
        self.data = self._read_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
            
        return torch.tensor(sample, dtype=torch.float32).to(device)

    # inner function
    def _info(self) -> tuple:
        return len(self.data), self.data[0].shape

    def _read_data(self) -> list:
        files = [file for file in os.listdir(self.dir_files) if file.endswith('.mat')]
        dataset = []

        for file in files:
            data = loadmat(os.path.join(self.dir_files, file))['EEG'][0][0][15][:63] # 63 - EEG channels
            dataset.append(data)

            print(f"{file} added")
        
        return dataset
    
    def save(self) -> None:
        pickle.dump(self, open(f"{self.save_path}/dataset_EEG.pkl", 'wb'), True)

        print('dataset has been saved')

    @staticmethod
    def load_dataset(file_name: str) -> list:
        data = pickle.load(open(file_name,'rb'))
        
        print('dataset has been loaded')
        return data


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = EEGDataset(dir_files='./data/d002')
    data.save()

    # data = EEGDataset.load_dataset("./data/dataset_EEG.pkl")
    # print(data._info())
    
    print('0')
