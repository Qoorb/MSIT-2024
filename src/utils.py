import os

from torch.utils.data import DataLoader
import torch

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from dataset import EEGDataset


def plotting(model, valid_loader, device):
    
    model.eval()
    
    points = []
    path = "./data/reconstructed_plots"
    
    for data in valid_loader:
        inputs = data.to(device) # [d.to(device) for d in data]

        outputs = model(inputs)
        points.extend(outputs.squeeze().detach().cpu().numpy())

        del inputs, outputs
    
    points = np.array(points)
    df = pd.DataFrame(points[0])
    df.to_csv('./data/figure_1_model.csv')
    

    # plot(points, path)


def plot(data, path='./data/plot') -> None:

    if not os.path.exists(path): os.mkdir(path)
    data = [d.squeeze().detach().cpu().numpy() for d in data]
    
    df = pd.DataFrame(data[0][0])
    df.to_csv('./data/figure_1_data.csv')

    # for i in range(len(data)):
    #     for j in range(63):
    #         plt.plot(data[i][j])

    #         plt.title('Lines Plot of a data')
    #         plt.xlabel('x-axis')
    #         plt.ylabel('y-axis')

    #         plt.savefig(f"{path}/plot_{i}_{j}.png")
    #         plt.close()


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Project running on device: ", device)
    
    model = torch.load("./model/autoencoder")
    valid_data = EEGDataset.load_dataset('./data/test_dataset_EEG.pkl')
    valid_loader = DataLoader(valid_data, batch_size=10, shuffle=False)

    plot(valid_loader)

    plotting(model, valid_loader, device)
