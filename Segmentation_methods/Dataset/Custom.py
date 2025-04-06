import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split


COLOR_MAP: Dict[Tuple[int, int, int], int] = {(0, 0, 1): 2, (1, 0, 0): 0, (0, 1, 0): 1}
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {2: (0, 0, 1), 0: (1, 0, 0), 1: (0, 1, 0)}


# ===========================
# Mapping between RGB colors and class indices
# ===========================
def rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Converts an RGB mask to a class index mask.

    @param mask_rgb: RGB mask of shape (H, W, 3)
    @return: Class index mask of shape (H, W)
    """
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.int8)
    for color, class_id in COLOR_MAP.items():
        match = np.all(mask_rgb == color, axis=-1)
        class_mask[match] = class_id
    return class_mask


def class_to_rgb(mask_class: np.ndarray) -> np.ndarray:
    """
    Converts a class index mask to an RGB mask.

    @param mask_class: Class index mask of shape (H, W)
    @return: RGB mask of shape (H, W, 3)
    """
    h, w = mask_class.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb_mask[mask_class == class_id] = color
    return rgb_mask




# ===========================
# Custom dataset class
# ===========================
class Custom(Dataset):
    def __init__(self, image_pkl_path: str, mask_pkl_path: str, transform: Optional[Any] = None):
        # Charge les images
        with open(image_pkl_path, "rb") as f:
            self.images = pickle.load(f).astype("float32")  # (N, H, W, 3)

        # Charge les masques (en RGB)
        with open(mask_pkl_path, "rb") as f:
            self.masks = pickle.load(f).astype("float32")  # (N, H, W, 3)

        self.images = self.images.transpose((0, 3, 1, 2))  # (N, C, H, W)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])  # (C, H, W)
        mask_rgb = self.masks[idx]  # (H, W, 3)
        mask_class = rgb_to_class(mask_rgb)  # (H, W)

        mask_rgb_tensor = torch.from_numpy(mask_rgb)  # pour affichage
        mask_class_tensor = torch.from_numpy(mask_class).long()  # pour entraînement

        if self.transform:
            image = self.transform(image)

        return image, mask_rgb_tensor, mask_class_tensor
    


# ===========================
# Function to split the data 
# 80% for training the rest for testing
# ===========================    
def split_data(images_path: str, masks_path: str, save_dir: str) -> None:
    """
    Splits the dataset into training (80%) and testing (20%) sets, and saves them.

    @param images_path: Path to the pickle file containing the images
    @param masks_path: Path to the pickle file containing the masks
    @param save_dir: Directory to save the train/test split pickle files
    """
    with open(images_path, "rb") as f:
        X = pickle.load(f)

    with open(masks_path, "rb") as f:
        Y = pickle.load(f)

    print("Original shape:", X.shape, Y.shape)

    # Split the dataset (with fixed random seed for reproducibility)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    with open(os.path.join(save_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)

    with open(os.path.join(save_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)

    with open(os.path.join(save_dir, "Y_train.pkl"), "wb") as f:
        pickle.dump(Y_train, f)

    with open(os.path.join(save_dir, "Y_test.pkl"), "wb") as f:
        pickle.dump(Y_test, f)

    print("Train shape:", X_train.shape, Y_train.shape)
    print("Test shape:", X_test.shape, Y_test.shape)
