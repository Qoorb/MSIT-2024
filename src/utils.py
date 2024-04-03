import os

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from dataset import EEGDataset


def plotting(model, valid_loader, device):
    
    model.eval()
    
    points = []
    path = "./data/reconstructed_plots"
    
    for data in valid_loader:
        inputs = data.to(device) # [d.to(device) for d in data]

        outputs = model(inputs)
        points.extend(outputs.detach().numpy())

        del inputs, outputs
    
    points = np.array(points)

    plot(points, path)


def plot(data, path='./data/plot') -> None:

    if not os.path.exists(path): os.mkdir(path)
    data = [d for d in data]
    
    for i in range(len(data)):
        for j in range(64):
            plt.plot(data[i][0][j])

            plt.title('Lines Plot of a data')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')

            plt.savefig(f"{path}/plot/plot_{i}_{j}.png")
            plt.close()


if __name__ == '__main__':
    
    valid_data = EEGDataset.load_dataset('./data/test_dataset_EEG.pkl')
    valid_loader = DataLoader(valid_data, batch_size=10, shuffle=True)

    plot(valid_loader)
