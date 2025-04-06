import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import torch
import numpy as np
from Segmentation_methods.Dataset.Custom import class_to_rgb


# ===========================
# PLot data input image
# ===========================
def show_sample(image: torch.Tensor, mask: np.ndarray, index: int = 0) -> None:
    """
    Displays a sample image and its corresponding mask side by side.

    @param image: Image tensor of shape (C, H, W)
    @param mask: Corresponding mask as a NumPy array (H, W)
    @param index: Index of the sample (for display purposes)
    """
    image_np = image.numpy().transpose(1, 2, 0)  # Convert (C, H, W) → (H, W, C)
    mask_np = mask

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image_np)
    ax[0].set_title(f"Image {index}")
    ax[0].axis("off")

    ax[1].imshow(mask_np, cmap="gray")
    ax[1].set_title(f"Mask {index}")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


# ===========================
# Apply inference on a sample with
# pretrained model
# ===========================
def inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
) -> None:
    """
    Performs inference on one sample from the test loader and displays the original image,
    the ground truth mask, and the predicted mask.

    @param model: Trained segmentation model
    @param test_loader: DataLoader containing the test dataset
    @param device: Torch device (e.g., torch.device("cuda") or torch.device("cpu"))
    @param save_path: Path to the saved model weights (.pth file)
    """
    # Load saved model weights
    model.load_state_dict(torch.load(save_path, map_location=device))

    model.eval()
    image, mask_rgb, mask_true = next(
        iter(test_loader)
    )  # image: [1, 3, H, W], mask_rgb: [1, H, W, 3], mask_true: [1, H, W]

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)  # Output shape: [1, num_classes, H, W]
        preds = torch.argmax(output, dim=1)  # Predicted class mask: [1, H, W]

    # Convert tensors to numpy arrays for visualization
    image_np = image[0].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]

    mask_pred = preds[0].cpu().numpy()  # [H, W]
    mask_pred_rgb = class_to_rgb(mask_pred)  # [H, W, 3]

    # Display original image, ground truth, and prediction
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb[0])
    axes[1].set_title("Ground Truth Mask (GT)")
    axes[1].axis("off")

    axes[2].imshow(mask_pred_rgb * 255)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
