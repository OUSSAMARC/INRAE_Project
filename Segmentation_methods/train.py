import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import segmentation_models_pytorch as smp
from Dataset import Custom  
import os


def train_model(model, train_loader, val_loader, device, optimizer, criterion, save_path, num_epochs=20, lr=1e-3):

    best_val_loss = float('inf') # Biggest value
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Phase entraînement
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
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Phase validation
        val_loss, mean_iou = validate_model(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Mean IoU: {mean_iou:.4f}")

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ious = []

    with torch.no_grad():
        for images, _, masks in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou = compute_mean_iou(preds, masks, num_classes=3)
            ious.append(iou)

    return total_loss / len(val_loader), sum(ious) / len(ious)


def compute_mean_iou(preds, masks, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Ignore cette classe si elle est absente
        else:
            ious.append(intersection / union)
    # Ne pas compter les NaN
    ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    return sum(ious) / len(ious) if ious else 0.0