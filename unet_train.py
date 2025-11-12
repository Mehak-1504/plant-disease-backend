import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

# ========================
# 1. Dataset
# ========================
class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(root, f))
        if len(self.images) == 0:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        rel_path = os.path.relpath(img_path, self.img_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        mask_candidates = [
            os.path.join(self.mask_dir, rel_dir, base_name + ".jpg"),
            os.path.join(self.mask_dir, rel_dir, base_name + ".png"),
            os.path.join(self.mask_dir, rel_dir, base_name + "_final_masked.jpg"),
            os.path.join(self.mask_dir, rel_dir, base_name + "_final_masked.png"),
            os.path.join(self.mask_dir, rel_dir, base_name + "_mask.jpg"),
            os.path.join(self.mask_dir, rel_dir, base_name + "_mask.png")
        ]

        # Try to find the mask
        mask_path = None
        for path in mask_candidates:
            if os.path.exists(path):
                mask_path = path
                break

        # If mask is missing or unreadable, retry another image
        if mask_path is None or not os.path.exists(mask_path):
            new_idx = np.random.randint(0, len(self.images))
            return self.__getitem__(new_idx)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            new_idx = np.random.randint(0, len(self.images))
            return self.__getitem__(new_idx)

        # Resize smaller for speed
        image = cv2.resize(image, (96, 96))
        mask = cv2.resize(mask, (96, 96))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)


# ========================
# 2. U-Net
# ========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dconv_up3 = DoubleConv(512, 256)
        self.dconv_up2 = DoubleConv(256, 128)
        self.dconv_up1 = DoubleConv(128, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return self.conv_last(x)


# ========================
# 3. Dice Loss
# ========================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


# ========================
# 4. Validation
# ========================
def validate(model, loader, device):
    model.eval()
    preds, gts = [], []
    criterion = nn.BCEWithLogitsLoss()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            pred = (torch.sigmoid(outputs) > 0.5).float()
            preds.append(pred.cpu().numpy())
            gts.append(masks.cpu().numpy())
            correct += (pred == masks).float().sum().item()
            total += masks.numel()

    acc = correct / total
    preds = np.concatenate(preds).astype(int).flatten()
    gts = np.concatenate(gts).astype(int).flatten()
    dice = (2. * np.sum(preds * gts)) / (np.sum(preds) + np.sum(gts) + 1e-7)
    return val_loss / len(loader), acc, dice, gts, preds


# ========================
# 5. Training Loop
# ========================
def train_model(model, train_loader, val_loader, device, epochs=2, save_path="best_unet_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    log_file = open("training_results.txt", "w", encoding="utf-8")

    print("\n🧠 Training Started...\n")
    log_file.write("🧠 Training Log\n\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")

        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == masks).float().sum().item()
            total += masks.numel()
            acc = correct / total
            loop.set_postfix(loss=loss.item(), acc=acc)

        val_loss, val_acc, val_dice, gts, preds = validate(model, val_loader, device)
        summary = (
            f"Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_loader):.4f} | "
            f"Acc: {acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f} | Dice: {val_dice:.4f}\n"
        )
        print(summary)
        log_file.write(summary)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved.\n")
            log_file.write("✅ Best model saved.\n")

    # Final evaluation
    model.load_state_dict(torch.load(save_path))
    val_loss, val_acc, val_dice, gts, preds = validate(model, val_loader, device)

    prec = precision_score(gts, preds, zero_division=0)
    rec = recall_score(gts, preds, zero_division=0)
    f1 = f1_score(gts, preds, zero_division=0)

    cm = confusion_matrix(gts, preds)
    cr = classification_report(gts, preds, target_names=["Background", "Leaf/ROI"])

    result_text = (
        f"\nAverage Dice Score: {val_dice:.4f}\n"
        f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}\n\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Classification Report:\n{cr}\n"
    )
    print(result_text)
    log_file.write(result_text)
    log_file.close()
    print("\n✅ Results saved in 'training_results.txt'\n")


# ========================
# 6. Main
# ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = LeafDataset("dataset/images", "dataset/masks")
    val_dataset = LeafDataset("dataset/val_images", "dataset/val_masks")

    # Smaller subset for speed (optional)
    train_dataset.images = train_dataset.images[:200]
    val_dataset.images = val_dataset.images[:50]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet().to(device)
    train_model(model, train_loader, val_loader, device, epochs=2, save_path="best_unet_model.pth")
