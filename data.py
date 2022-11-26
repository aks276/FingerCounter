import torch

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import cv2

import os

import config

"""Dataset downloaded from Kaggle Fingers competition"""

class FingerCountingDataset(Dataset):
    def __init__(self, train_dir, transforms=None):
        self.train_dir = train_dir
        self.transforms = transforms
        self.img_list = os.listdir(self.train_dir)

    def __len__(self):
        return len(os.listdir(self.train_dir))

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = cv2.imread(os.path.join(self.train_dir, image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        number = image_path[-6]
        # output = torch.zeros(6)
        # output[int(number)] = 1
        output = int(number)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, output


train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=512),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


train_dataset = FingerCountingDataset(train_dir=config.TRAIN_PATH, transforms=train_transform)
test_dataset = FingerCountingDataset(train_dir=config.TEST_PATH, transforms=test_transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)