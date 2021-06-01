import os
import torch
import torchvision
import pandas as pd

from PIL import Image as pil
from PIL import ImageOps as pilops
from sklearn import model_selection

ROOT_DATAPATH = '../dataset/covidx-cxr2'


class CXR2Dataset(torch.utils.data.Dataset):
    def __init__(self, path, dataframe, equalize=False, augment=False):
        self.path = path
        self.equalize = equalize

        # Extract the filenames and classes from the dataframe
        self.filenames = dataframe['filename'].to_numpy()
        self.targets = dataframe['class'].to_numpy()

        # Initialize data transforms
        transforms = [torchvision.transforms.ToTensor()]
        if augment:
            transforms.append(torchvision.transforms.RandomAffine(
                degrees=10.0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR  # use bilinear interpolation
            ))
        transforms.append(torchvision.transforms.Normalize((0.5,), (0.5,)))  # normalize to (-1, 1)
        self.transform = torchvision.transforms.Compose(transforms)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        # Obtain the filename and target
        filename = self.filenames[i]
        target = self.targets[i]

        # Load and transform the image
        with pil.open(os.path.join(self.path, filename)) as img:
            if self.equalize:
                img = pilops.equalize(img)
            data = self.transform(img)
        return data, target

    def get_targets(self):
        return self.targets


def load_train_valid_datasets(valid_size=0.1, equalize=False, augment=True, random_state=1816):
    # Load the train dataframe
    filepath = os.path.join(ROOT_DATAPATH, 'train', 'train.csv')
    df = pd.read_csv(filepath, usecols=['filename', 'class'])

    # Split the dataframe in train and validation
    train_df, valid_df = model_selection.train_test_split(
        df, test_size=valid_size, shuffle=True, stratify=df['class'], random_state=random_state
    )

    # Instantiate the datasets (notice data augmentation on train data)
    images_path = os.path.join(ROOT_DATAPATH, 'train', 'images')
    train_data = CXR2Dataset(images_path, train_df, equalize=equalize, augment=augment)
    valid_data = CXR2Dataset(images_path, valid_df, equalize=equalize, augment=False)
    return train_data, valid_data


def load_test_dataset(equalize=False):
    # Load the test dataframe
    filepath = os.path.join(ROOT_DATAPATH, 'test', 'test.csv')
    test_df = pd.read_csv(filepath, usecols=['filename', 'class'])

    # Instantiate the dataset
    images_path = os.path.join(ROOT_DATAPATH, 'test', 'images')
    test_data = CXR2Dataset(images_path, test_df, equalize=equalize, augment=False)
    return test_data
