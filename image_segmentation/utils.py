import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics import Accuracy
from torchmetrics.segmentation import MeanIoU

import matplotlib.pyplot as plt


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    iou = MeanIoU(num_classes=2).to(device)
    acc = Accuracy("binary").to(device)

    with torch.no_grad():
        for img, target in loader:
            img = img.to(device)
            target = target.to(device)

            output = model(img)
            probs = F.softmax(output, dim=1)
            preds = F.one_hot(probs.argmax(dim=1), num_classes=3).permute(0, 3, 1, 2)
            iou(preds, target)  # ! update() for MeanIOU doesn't work correctly in v1.4.1 (#2698)
            acc.update(preds, target)

    return iou.compute().item(), acc.compute().item()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logger = {
        "train": {"loss": [], "iou": [], "acc": []},
        "val": {"iou": [], "acc": []},
    }

    for i in range(num_epochs):
        model.train()

        # Training loop
        epoch_loss = 0.0
        for img, target in train_loader:
            img = img.to(device)
            target = target.type(torch.float).to(device)

            output = model(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # Evaluate on train and validation set
        train_iou, train_acc = evaluate(model, train_loader, device)
        val_iou, val_acc = evaluate(model, val_loader, device)

        # Add metrics to logger
        logger["train"]["iou"].append(train_iou)
        logger["train"]["acc"].append(train_acc)
        logger["val"]["iou"].append(val_iou)
        logger["val"]["acc"].append(val_acc)

        print(
            f"{(i+1):{len(str(num_epochs))}}/{num_epochs}: {train_loss=:.4f}, {train_iou=:.4f}, {val_iou=:.4f}, {train_acc=:.4f}, {val_acc=:.4f}"
        )

    return logger


@torch.no_grad()
def infer(model, img: torch.Tensor, device: str) -> torch.Tensor:
    logits = model(img.to(device))
    probs = F.softmax(logits, dim=1)
    mask = probs.argmax(1, keepdim=True).cpu()

    return mask


def plot_examples(img, target_mask, pred_mask):
    num_images = img.size(0)
    _, axs = plt.subplots(num_images, 3, figsize=(3 * 3, num_images * 3))

    axs[0, 0].set_title("Image", fontsize=10)
    axs[0, 1].set_title("Truth", fontsize=10)
    axs[0, 2].set_title("Prediction", fontsize=10)

    for i, tensor in enumerate([img, target_mask, pred_mask]):
        tensor = tensor.permute(0, 2, 3, 1)  # (N, C, H, W) => (N, H, W, C)

        for j in range(num_images):
            axs[j, i].imshow(tensor[j])
            axs[j, i].axis("off")

    plt.show()
