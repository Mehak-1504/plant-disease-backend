import os
import io
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import requests

# ========================
# 1. Flask App & CORS
# ========================
app = Flask(__name__)
CORS(app)

# ========================
# 2. U-Net Model Definition
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
# 3. Download Model Automatically
# ========================
MODEL_URL = "https://github.com/Mehak-1504/plant-disease-backend/releases/download/v1.0/best_unet_model.pth"
MODEL_PATH = "best_unet_model.pth"

if not os.path.exists(MODEL_PATH):
    print("⚠️ Model file not found — downloading...")
    r = requests.get(MODEL_URL)
    open(MODEL_PATH, "wb").write(r.content)
    print("✅ Downloaded best_unet_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded correctly!")
except Exception as e:
    print("❌ Error loading model:", e)

# ========================
# 4. Image Transform
# ========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ========================
# 5. Prediction Route
# ========================
@app.route('/api/detect', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).cpu().squeeze().numpy()

    # Convert mask to uint8 image
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(pred_mask).convert("L")

    # Convert mask to bytes
    byte_io = io.BytesIO()
    mask_img.save(byte_io, 'PNG')
    byte_io.seek(0)

    # Example JSON data (replace with actual disease + remedy logic later)
    result_json = {
        "disease": "Example Leaf Disease",
        "remedy": "Apply neem oil and avoid overwatering"
    }

    # Return mask image as file and JSON as headers (or return JSON + mask URL)
    # For now, just send mask image
    return send_file(byte_io, mimetype='image/png')

# ========================
# 6. Run Server
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
