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
        # self.files = [file for file in os.listdir(dir_files) if file.endswith('.mat')]
        # self.data = self._read_data()
        self.data = []
    
    # inner function
    def _read_data(self) -> list:
        dataset = []

        for file in self.files:
            data = loadmat(os.path.join(self.dir_files, file))['EEG'][0][0][15][:64] # 64 - EEG channels
            dataset.append(np.array(data))

            print(f"{file} added")
        
        return dataset
    
    def save(self) -> None:
        pickle.dump(self, open(f"{self.save_path}/dataset_EEG.pkl", 'wb'), True)
        # np.savetxt(f"{self.save_path}/dataset_EEG.csv", self.data, delimiter=",", fmt='%.3f')

        print('dataset has been saved')

    @staticmethod
    def load_data(file_name: str) -> list:
        data = pickle.load(open(file_name,'rb'))
        # self.data = np.loadtxt(file_name)
        
        print('dataset has been loaded')
        return data


if __name__ == '__main__':
    # data = EEGDataset(dir_files='./data/d002')
    # data.save()

    # data = pickle.load(open("data/dataset_EEG.pkl",'rb'))
    # print(data.data)

    data = EEGDataset().load_data("data/dataset_EEG.pkl")
    print(data.data)
    print('0')
