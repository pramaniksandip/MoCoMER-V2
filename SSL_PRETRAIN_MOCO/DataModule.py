from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DataModule(Dataset):
    def __init__(self, bs, root_path, df, cv2_transforms, pil_transforms):
        self.bs = bs
        self.root_path = root_path
        self.cv2_transforms = cv2_transforms
        self.pil_transforms = pil_transforms
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Image']
        img_path = self.root_path + img_path
        image = Image.open(img_path)
        image = np.array(image)


        orgpic1, orgpic2 = self.__augment__(image)
        return orgpic1, orgpic2

    def __augment__(self, x):
        x = self.cv2_transforms(x)  # Apply OpenCV transformations to NumPy array
        x = Image.fromarray(x)  # Convert NumPy array to PIL Image
        x1, x2 = self.pil_transforms(x), self.pil_transforms(x)  # Apply PIL transformations
        return x1, x2

def collate_fn(batch):
    orgpic1_batch = [item[0] for item in batch]
    orgpic2_batch = [item[1] for item in batch]

    heights_x1 = [s.size(1) for s in orgpic1_batch]
    widths_x1 = [s.size(2) for s in orgpic1_batch]
    heights_x2 = [s.size(1) for s in orgpic2_batch]
    widths_x2 = [s.size(2) for s in orgpic2_batch]

    n_samples = len(orgpic1_batch)
    max_height_x1 = max(heights_x1)
    max_width_x1 = max(widths_x1)
    max_height_x2 = max(heights_x2)
    max_width_x2 = max(widths_x2)

    max_height = max(max_height_x1, max_height_x2)
    max_width = max(max_width_x1, max_width_x2)

    x1 = torch.zeros(n_samples, 1, max_height, max_width)
    x2 = torch.zeros(n_samples, 1, max_height, max_width)
    img_mask = torch.ones(n_samples, max_height, max_width, dtype=torch.bool)

    for idx, (s_x1, s_x2) in enumerate(zip(orgpic1_batch, orgpic2_batch)):
        x1[idx, :, :heights_x1[idx], :widths_x1[idx]] = s_x1
        x2[idx, :, :heights_x2[idx], :widths_x2[idx]] = s_x2
        img_mask[idx, :max(heights_x1[idx], heights_x2[idx]), :max(widths_x1[idx], widths_x2[idx])] = 0



    return x1, x2, img_mask

class Batch:
    def __init__(self, x1, x2, img_mask):
        self.x1 = x1
        self.x2 = x2
        self.img_mask = img_mask
