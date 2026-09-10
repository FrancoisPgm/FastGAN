import argparse
import os

import torch
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


# ==========================================
# 1. Custom Dataset Class
# ==========================================
class FolderDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(data_dict.keys())}

        for cls_name, folder_path in data_dict.items():
            label = self.class_to_idx[cls_name]
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found. Skipping.")
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.images.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# 2. Model Initialization
# ==========================================
def init_weights(m):
    """Applies He (Kaiming) Initialization to Conv and Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_resnet18(num_classes):
    # Load architecture without weights
    model = models.resnet18(weights=None)

    # Apply He Initialization
    model.apply(init_weights)

    # Replace only the final FC layer (No Dropout added here)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==========================================
# 3. Training Logic
# ==========================================
def train_advanced_resnet(data_dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = FolderDataset(data_dict, transform=train_transform)
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please check your data dictionary paths.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = get_resnet18(len(data_dict)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Warmup + Cosine Decay
    warmup_sch = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_sch = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warmup_epochs))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[args.warmup_epochs]
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {running_loss / len(train_loader):.4f} - Val Acc: {100 * correct / total:.2f}%"
        )

    return model


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet18 from scratch with advanced optims."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Folder to save model and labels",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="resnet18_model.pth",
        help="Name of the saved model file",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    # Your data dictionary
    my_data = {"cats": "./data/cats", "dogs": "./data/dogs", "birds": "./data/birds"}

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Train the model
    trained_model = train_advanced_resnet(my_data, args)

    # Save Model Weights
    model_path = os.path.join(args.output_folder, args.output_name)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save Labels to text file
    labels_path = os.path.join(args.output_folder, "labels.txt")
    with open(labels_path, "w") as f:
        f.writelines(f"{idx}: {label}\n" for idx, label in enumerate(my_data.keys()))
    print(f"Labels saved to {labels_path}")
