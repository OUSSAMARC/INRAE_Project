import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import segmentation_models_pytorch as smp
from Dataset import Custom  
import os
import matplotlib.pyplot as plt


# ===========================
# Function to train the model
# ===========================
def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, save_path: str, num_epochs: int = 20, lr: float = 1e-3) -> None:
    """
    Trains the segmentation model and validates it on each epoch.

    @param model: The segmentation model to train
    @param train_loader: DataLoader for the training dataset
    @param val_loader: DataLoader for the validation dataset
    @param device: The device to use (e.g., "cuda" or "cpu")
    @param save_path: File path to save the best model weights
    @param num_epochs: Number of training epochs
    @param lr: Learning rate for the optimizer
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for images, _, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.6f}")

        # Validation phase
        val_loss, mean_jaccard, mean_dice, mean_Recall, mean_Precision, mean_ConfIndex = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.6f} | Mean Jaccard: {mean_jaccard:.6f} | Mean Dice: {mean_dice:.6f} | Mean Recall: {mean_Recall} | Mean Precision: {mean_Precision} | Mean ConfIndex: {mean_ConfIndex}")
        
        plot_train_val_loss(train_losses, val_losses)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)


# ===========================
# Function to evaluate the model
# ===========================
def validate_model(model: torch.nn.Module, val_loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> tuple:
    """
    Validates the model on the validation set and computes metrics.

    @param model: Trained segmentation model
    @param val_loader: DataLoader for validation data
    @param criterion: Loss function
    @param device: The device to use (e.g., "cuda" or "cpu")
    @return: Tuple containing average loss and mean metrics (Jaccard, Dice, Recall, Precision, ConfIndex)
    """
    model.eval()
    total_loss = 0.0
    Jaccard = []
    dices = []
    R = []
    P = []
    C = []

    with torch.no_grad():
        for images, _, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            jaccard, dice, recall, precision, confindex = compute_mean_iou(preds, masks, num_classes=3)
            Jaccard.append(jaccard)
            dices.append(dice)
            R.append(recall)
            P.append(precision)
            C.append(confindex)

    return total_loss / len(val_loader), sum(Jaccard) / len(Jaccard), sum(dices) / len(dices), sum(R) / len(R), sum(P) / len(P), sum(C) / len(C)


# ===========================
# Indexes for evaluation
# ===========================
def compute_mean_iou(preds: torch.Tensor, masks: torch.Tensor, num_classes: int) -> tuple:
    """
    Computes per-class and mean metrics for segmentation output.

    @param preds: Predicted class masks of shape (B, H, W)
    @param masks: Ground truth masks of shape (B, H, W)
    @param num_classes: Total number of segmentation classes
    @return: Tuple of mean Jaccard, Dice, Recall, Precision, and Confusion Index
    """
    Jaccard = []
    dice = []
    Recalls = []
    Precisions = []
    ConfmIndexs = []
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls
        FP = (pred_inds & (~target_inds)).sum()
        FN = (target_inds & (~pred_inds)).sum()
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        Sum = pred_inds.sum().item() + target_inds.sum().item()
        Recall = intersection / (intersection + FN)
        Precision = intersection / (intersection + FP)
        ConfmIndex = 1 - (FP + FN) / (intersection)
        
        Recalls.append(Recall)
        Precisions.append(Precision)
        ConfmIndexs.append(ConfmIndex)
        dice.append(2 * intersection / Sum)
        Jaccard.append(intersection / union)
    
    return sum(Jaccard) / len(Jaccard), sum(dice) / len(dice), sum(Recalls) / len(Recalls), sum(Precisions) / len(Precisions), sum(ConfmIndexs) / len(ConfmIndexs)


# ===========================
# PLot the train and validation losses
# ===========================
def plot_train_val_loss(train_losses: list, val_losses: list) -> None:
    """
    Plots training and validation loss over epochs.

    @param train_losses: List of training loss values
    @param val_losses: List of validation loss values
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linestyle="--")
    plt.plot(epochs, val_losses, label="Validation Loss", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    