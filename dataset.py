import os

import pickle

import numpy as np

from torch.utils.data import Dataset

from scipy.io import loadmat


class EEGDataset(Dataset):
    def __init__(self, save_path='./data', dir_files='', transform=None) -> None:
        self.save_path = save_path
        self.dir_files = dir_files
        self.transform = transform
        self.files = [file for file in os.listdir(dir_files) if file.endswith('.mat')]
        self.data = self.load_data()
    
    def load_data(self) -> list:
        dataset = []

        for file in self.files:
            data = loadmat(os.path.join(self.dir_files, file))['EEG'][0][0][15]
            dataset.append(np.array(data))

            print(f"{file} added")
        
        return dataset
    
    def save(self) -> None:
        pickle.dump(self, open(f"{self.save_path}/dataset_EEG.pkl", 'wb'), True)
        
        print('dataset has been saved')




if __name__ == '__main__':
    data = EEGDataset(dir_files='./data/d002')
    print(data.data)

    print('0')
