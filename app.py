import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from flask import Flask, request, send_file

app = Flask(__name__)
CORS(app)

# ========================
# 1. U-Net Model (same as in unet_train.py)
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
# 2. Flask Setup
# ========================
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = UNet().to(device)
model_path = "../best_unet_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found! Train first or check the path.")


# ========================
# 3. Image Transform
# ========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ========================
# 4. Prediction Endpoint
# ========================
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).cpu().squeeze().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Convert mask to list (for frontend JSON)
    mask_list = pred_mask.tolist()

    return jsonify({
        'message': 'Prediction successful',
        'mask': mask_list
    })


# ========================
# 5. Run Server
# ========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
