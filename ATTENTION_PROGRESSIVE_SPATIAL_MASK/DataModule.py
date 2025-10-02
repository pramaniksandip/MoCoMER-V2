# DataModule

from PIL import Image
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import multi_hot_encode

char_to_index = {}
with open('/workspace/nemo/data/RESEARCH/HMER_MASK_MATCH_HME100K/HME100K/dictionary.txt', 'r') as file:
    for line in file:
        char, idx = line.strip().split('\t')
        char_to_index[char] = int(idx) - 1

fp=open("/workspace/nemo/data/RESEARCH/HMER_MASK_MATCH_HME100K/HME100K/train_labels.txt",'r')
labels=fp.readlines()
fp.close()

targets={}
    # map word to int with dictionary
for l in labels:
  tmp=l.strip().split()
  uid=tmp[0]
  cap = tmp[1]
  targets[uid]=cap

class ScaleToLimitRange:
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            img = cv.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv.INTER_LINEAR)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            img = cv.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv.INTER_LINEAR)
            return img

        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img

class ScaleAugmentation:
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv.resize(img, None, fx=k, fy=k, interpolation=cv.INTER_LINEAR)
        return img

class NumpyToPIL:
    def __call__(self, img: np.ndarray) -> Image.Image:
        return Image.fromarray(img)

class PILToNumpy:
    def __call__(self, img: Image.Image) -> np.ndarray:
        return np.array(img)



class DataModule(nn.Module):
    def __init__(self, bs, root_path, df, cv2_transforms, pil_transforms):
        super().__init__()
        self.bs = bs
        self.root_path = root_path
        self.cv2_transforms = cv2_transforms
        self.pil_transforms = pil_transforms
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Image']
#         img_c = img_path.replace(".jpg","")
        img_c = img_path
        caption = targets[img_c]
        img_path = self.root_path + img_path


        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        label = multi_hot_encode(caption,char_to_index)
        x_mask = torch.zeros(image.shape[0], image.shape[1])


        orgpic = self.__augment__(image)
        return orgpic, x_mask, label

    def __augment__(self, x):
        x = self.cv2_transforms(x)  # Apply OpenCV transformations to NumPy array
        x = Image.fromarray(x)  # Convert NumPy array to PIL Image

        x = self.pil_transforms(x)#, self.pil_transforms(x)  # Apply PIL transformations

        return x




def collate_fn(batch):
    orgpic_batch = [item[0] for item in batch]
    # x_mask_batch = [item[1] for item in batch
    label_batch = [item[2] for item in batch]


    heights_x = [s.size(1) for s in orgpic_batch]
    widths_x = [s.size(2) for s in orgpic_batch]
    # heights_x2 = [s.size(1) for s in orgpic2_batch]
    # widths_x2 = [s.size(2) for s in orgpic2_batch]

    n_samples = len(orgpic_batch)
    max_height = max(heights_x)
    max_width = max(widths_x)
    # max_height_x2 = max(heights_x2)
    # max_width_x2 = max(widths_x2)

    x = torch.zeros(n_samples, 3, max_height, max_width)
    # x2 = torch.zeros(n_samples, 1, max_height, max_width)
    img_mask = torch.ones(n_samples, max_height, max_width, dtype=torch.bool)

    for idx, (s_x) in enumerate(orgpic_batch):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x
        # x2[idx, :, :heights_x2[idx], :widths_x2[idx]] = s_x2
        img_mask[idx, :heights_x[idx], :widths_x[idx]] = 0



    return x, img_mask, torch.stack(label_batch)
class Batch:
    def __init__(self, orgpic, x_mask, label):
        self.orgpic = orgpic
        self.x_mask = x_mask
        self.label = label

