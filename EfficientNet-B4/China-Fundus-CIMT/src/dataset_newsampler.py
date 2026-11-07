import os
import json
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, WeightedRandomSampler
# equivalent to
# import torch
# Dataset = torch.utils.data.Dataset
# DataLoader = torch.utils.data.DataLoader
# WeightedRandomSampler = torch.utils.data.WeightedRandomSampler
from PIL import Image

# resizing the fundus images to 380*380 to be compatible with EfficientNet-B4
def get_transform(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomHorizontalFlip(p=0.5), # randomly flipping the images
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,  # vary brightness
            contrast=0.2,    # vary contrast
            saturation=0.2,  # vary color intensity
            hue=0.03         # vary hue slightly
        ),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.1),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, valid_transform

class FundusDataset(Dataset):
    def __init__(self, dataframe, img_dir, indices, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.indices = indices

    @classmethod
    def from_json(cls, json_dir, img_dir, group_value, transform=None):
        with open(str(json_dir), 'r') as f:
            data = json.load(f)
        dataframe = pd.DataFrame.from_dict(data, orient='index')
        filtered_dataframe = dataframe[dataframe['group'] == group_value]
        indices = filtered_dataframe.index.tolist()
        return cls(filtered_dataframe, img_dir, indices, transform)

    def __len__(self):
        return len(self.indices) # this ensures idx is between 0 and len(self.indices) - 1

    def __getitem__(self, idx):
        id = self.indices[idx]
        pair = self.dataframe.loc[id]
        left_path = os.path.join(self.img_dir, pair["left_eye"])
        right_path = os.path.join(self.img_dir, pair["right_eye"])
        left_image = Image.open(left_path).convert("RGB")
        right_image = Image.open(right_path).convert("RGB")
        label = pair["label"]

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        return left_image, right_image, label

def get_soft_balanced_sampler(dataset, softness, sigma, threshold = 0.9):
    df = dataset.dataframe

    label_counts = df['label'].value_counts().to_dict()

    class_weights = {cls: (1.0 / count) ** (1 - softness) for cls, count in label_counts.items()} # compute class (cls) weights as a python dictionary, ** put softness as exponential

    sample_class_weights = df['label'].map(class_weights).values  # map class weights to the 'diagnosis' labels

    cimt_values = df['thickness'].astype(float).values

    boundary_weights = np.exp(-((cimt_values - threshold) ** 2) / (2 * sigma ** 2))

    sample_weights = sample_class_weights * boundary_weights

    sampler = WeightedRandomSampler(weights=sample_weights, # generate the sampler
                                    num_samples=len(sample_weights), # a convenient choice, despite not all images are used, the total no of images trained in 1 epoch remains the same
                                    replacement=True)

    return sampler, class_weights
