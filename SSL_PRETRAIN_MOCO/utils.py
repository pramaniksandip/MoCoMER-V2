import os
import pandas as pd
import numpy as np
import cv2
import random
from PIL import ImageFilter, ImageOps
import torch
import torch.nn.functional as F

class DataPreperation():
    def __init__(self, root_path):
        self.root_path = root_path
        #self.files = []
        self.image = []
        #self.output = []
        #self.data_image = []
        #self.data_output = []

    def Train_Test_Data(self):
      count = 0
      print("Data Preperation Initiated!!!")

      for i in os.listdir(self.root_path):
        count+=1
        self.image.append(i)
        if (count % 500 == 0 ):
          print("Number of Data Prepared =====>{}".format(count))

      self.df_train = pd.DataFrame(list(zip(self.image)),columns =['Image'])

      return self.df_train
    
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
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
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
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

import random

def LLocal(x, x_plus, temperature=0.5, K=5):
    """
    Compute the local feature loss based on InfoNCE as described in the paper.

    Args:
        x: Tensor, feature map from the encoder (B, C, H, W)
        x_plus: Tensor, feature map from the momentum encoder (B, C, H, W)
        temperature: float, temperature scaling factor
        K: int, number of non-overlapping patches per side

    Returns:
        Loss: float, local feature loss
    """
    B, C, H, W = x.shape
    patch_size = H // K
    
    # Extract KxK patches from the feature maps
    patches_x = torch.stack([
        F.adaptive_avg_pool2d(x[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size], 1).squeeze(-1).squeeze(-1)
        for i in range(K) for j in range(K)
    ], dim=1)  # Shape: (B, K^2, C)

    patches_x_plus = torch.stack([
        F.adaptive_avg_pool2d(x_plus[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size], 1).squeeze(-1).squeeze(-1)
        for i in range(K) for j in range(K)
    ], dim=1)  # Shape: (B, K^2, C)

    # Compute positive pairs (i.e., matching patches between x and x_plus)
    pos_pairs = torch.einsum('bkc,bkc->bk', patches_x, patches_x_plus)  # Shape: (B, K^2)

    # Compute similarities to all patches (including negatives)
    all_pairs = torch.einsum('bkc,bjc->bkj', patches_x, patches_x)  # Shape: (B, K^2, K^2)

    # Mask out self-similarities in the negative pairs
    mask = ~torch.eye(K**2, device=x.device).bool()  # Shape: (K^2, K^2)
    neg_pairs = all_pairs[:, mask].view(B, K**2, -1)  # Shape: (B, K^2, K^2-1)

    # Concatenate positive and negative pairs for logits
    logits = torch.cat([pos_pairs.unsqueeze(-1), neg_pairs], dim=-1) / temperature  # Shape: (B, K^2, K^2)

    # Labels for cross-entropy: positive is the first logit in each set
    labels = torch.zeros(B * K**2, dtype=torch.long, device=x.device)

    # Compute the loss
    logits = logits.view(-1, logits.shape[-1])  # Shape: (B * K^2, K^2)
    return F.cross_entropy(logits, labels)
