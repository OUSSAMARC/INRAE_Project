import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

COLOR_MAP = {(0, 0, 255): 2, (255, 0, 0): 0, (0, 255, 0): 1}


def rgb_to_class(mask_rgb):
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.int8)
    for color, class_id in COLOR_MAP.items():
        match = np.all(mask_rgb == color, axis=-1)
        class_mask[match] = class_id
    return class_mask


class Custom(Dataset):
    def __init__(self, image_pkl_path, mask_pkl_path, transform=None):
        # Charge les images
        with open(image_pkl_path, "rb") as f:
            self.images = pickle.load(f).astype("float32")  # (N, H, W, 3)

        # Charge les masques (en RGB)
        with open(mask_pkl_path, "rb") as f:
            self.masks = pickle.load(f).astype("float32")  # (N, H, W, 3)

        self.images = self.images.transpose((0, 3, 1, 2))  # (N, C, H, W)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])  # (C, H, W)
        mask_rgb = self.masks[idx]  # (H, W, 3)
        mask_class = rgb_to_class(mask_rgb)  # (H, W)

        mask_rgb_tensor = torch.from_numpy(mask_rgb)  # pour affichage
        mask_class_tensor = torch.from_numpy(mask_class).long()  # pour entraînement

        if self.transform:
            image = self.transform(image)

        return image, mask_rgb_tensor, mask_class_tensor
