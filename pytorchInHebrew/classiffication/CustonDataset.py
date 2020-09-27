import cv2
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.image as img


class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, img_size=640):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_as_csv = pd.read_csv(csv_file)
        self.data_as_csv["class"] = pd.Categorical(pd.factorize(self.data_as_csv["class"])[0])
        self.labels = torch.zeros(pd.unique(self.data_as_csv["class"]).size)
        self.root_dir = root_dir
        self.transform = transform
        self.IMG_SIZE = img_size
        self.start = 0
        self.end = 0

    def __len__(self):
        return len(self.data_as_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.data_as_csv.iloc[idx, 2])
        image = img.imread(img_name)
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        label = self.labels.clone()
        label[self.data_as_csv.iloc[idx, 0]] = 1
        if self.transform:
            image = self.transform(image)
        return image, label


class SubSet(Dataset):
    def __init__(self, subset, transform=None, img_size=640):
        self.subset = subset
        self.transform = transform
        self.IMG_SIZE = img_size

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)