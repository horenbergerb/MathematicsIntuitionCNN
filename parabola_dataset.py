import os
import torch
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class ParabolaDataset(Dataset):

    def __init__(self, parabolas_dir, intercepts_dir, filenames_dir):
        self.parabolas_dir = parabolas_dir
        self.intercepts_dir = intercepts_dir
        self.filenames = pd.read_csv(filenames_dir)

        self.parabolas = []
        self.intercepts = []
        for filename in self.filenames.iloc[:, 0]:
            parabola_name = os.path.join(self.parabolas_dir, filename)
            intercept_name = os.path.join(self.intercepts_dir, filename)
            self.parabolas.append(torch.from_numpy(
                np.invert(np.array(
                    ImageOps.grayscale(Image.open(parabola_name)))).reshape((1, 120, 160)).astype(float)/255).float())
            self.intercepts.append(torch.from_numpy(
                np.invert(np.array(
                    ImageOps.grayscale(Image.open(intercept_name)))).reshape((1, 120, 160)).astype(float)/255).float()/(.89*9))
        # self.filenames = self.filenames.iloc[0:3000, :]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #parabola_name = os.path.join(self.parabolas_dir, self.filenames.iloc[idx, 0])
        #intercept_name = os.path.join(self.intercepts_dir, self.filenames.iloc[idx, 0])
        #parabola = torch.from_numpy(
        #    np.invert(np.array(
        #        ImageOps.grayscale(Image.open(parabola_name)))).reshape((1, 120, 160)).astype(float)/255).float()
        #intercept = torch.from_numpy(
        #    np.invert(np.array(
        #        ImageOps.grayscale(Image.open(intercept_name)))).reshape((1, 120, 160)).astype(float)/255).float()

        sample = {'parabola': self.parabolas[idx], 'intercept': self.intercepts[idx]}
        return sample


if __name__ == '__main__':
    parabola_ds = ParabolaDataset('parabolas', 'intercepts', 'filenames.csv')
    fig = plt.figure()
    for i in range(len(parabola_ds)):
        plt.imshow(np.where(parabola_ds[i]['parabola']
                            .numpy().reshape((120, 160)) > 0, 1, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()
        plt.imshow(np.where(parabola_ds[i]['intercept']
                            .numpy().reshape((120, 160)) > 0, 1, 0), cmap='gray', vmin=0, vmax=1)
        plt.show()
