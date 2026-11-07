import os
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
        transforms.RandomHorizontalFlip(p=0.5), # randomly flipping the images
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
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe) # this ensures idx is between 0 and len(self.dataframe) - 1

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
        return image, label

def get_soft_balanced_sampler(df, softness):

    label_counts = df['diagnosis'].value_counts().to_dict()

    class_weights = {cls: (1.0 / count) ** (1 - softness) for cls, count in label_counts.items()} # compute class (cls) weights as a python dictionary, ** put softness as exponential

    sample_weights = df['diagnosis'].map(class_weights).values # map class weights to the 'diagnosis' labels

    sampler = WeightedRandomSampler(weights=sample_weights, # generate the sampler
                                    num_samples=len(sample_weights), # a convenient choice, despite not all images are used, the total no of images trained in 1 epoch remains the same
                                    replacement=True)

    return sampler, class_weights
