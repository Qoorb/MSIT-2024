import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader

from dataset import EEGDataset
from models import Autoencoder
from models.processing import train, validate


def plotting(model, loader, device):
    
    model.eval()
    
    points = []
    label_idcs = []
    
    path = "./data/reconstructed_plots"
    if not os.path.exists(path): os.mkdir(path)
    
    for i, data in enumerate(loader):
        data, label = data.to(device) # [d.to(device) for d in data]

        data = model.encoder(data).to(device)
        points.extend(data.squeeze().detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())

        del data, label

    points = np.array(points)
    
    pca = PCA(n_components=2)
    compressed_data = pca.fit_transform(points)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x=compressed_data[:, 0], y=compressed_data[:, 1], s=2.0, 
                c=label_idcs, cmap='tab10', alpha=0.9, zorder=2)

    plt.savefig(f"{path}/latentspace.png", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Project running on device: ", device)

    model = torch.load('PATH')

    train_data = EEGDataset.load_dataset('./data/train_dataset_EEG.pkl')
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    plotting(model, train_loader, device)
