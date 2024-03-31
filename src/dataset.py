import os
import pickle

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from scipy.io import loadmat

import matplotlib.pyplot as plt


class EEGDataset(Dataset):
    def __init__(self, save_path='./data', dir_files='', transform=None, group_size=1000) -> None:
        self.save_path = save_path
        self.dir_files = dir_files
        self.transform = transform
        self.group_size = group_size
        self.data = self._read_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    # inner function
    def _info(self) -> tuple:
        return len(self.data), self.data[0].shape

    def _read_data(self) -> list:
        files = [file for file in os.listdir(self.dir_files) if file.endswith('.mat')]
        dataset = []

        for file in files[:6]:
            data = loadmat(os.path.join(self.dir_files, file))['EEG'][0][0][15][:64] # 64 - EEG channels
            
            for start_idx in range(0, data.shape[1], self.group_size):
                if start_idx + self.group_size <= data.shape[1]:
                    dataset.append(data[:, start_idx:start_idx + self.group_size])

            print(f"{file} added")
        
        return dataset
    
    def save(self) -> None:
        pickle.dump(self, open(f"{self.save_path}/test_dataset_EEG.pkl", 'wb'), True) # TODO: fix

        print('dataset has been saved')

    @staticmethod
    def load_dataset(file_name: str) -> object:
        data = pickle.load(open(file_name,'rb'))
        
        print('dataset has been loaded')
        return data


def plot(data, path='./data') -> None:

    if not os.path.exists('./data/plot'): os.mkdir('./data/plot')

    for i in range(len(data)):
        for j in range(64):
            plt.plot(data[i][0][j])

            plt.title('Lines Plot of a data')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')

            plt.savefig(f"{path}/plot/plot_{i}_{j}.png")
            plt.close()


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor() # transforms.Normalize(-1, 1)
    ])

    # data = EEGDataset(dir_files='./data/d002', transform=transform)
    # data.save()

    data = EEGDataset.load_dataset('./data/test_dataset_EEG.pkl')
    plot(data)
