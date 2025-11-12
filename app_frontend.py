import streamlit as st
import requests
from PIL import Image
import io

# Flask backend URL
BACKEND_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Leaf Disease Segmentation", layout="centered")

st.title("🌿 Leaf Disease Segmentation using U-Net")
st.write("Upload an image of a leaf and get the segmented mask using the trained U-Net model.")

uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Processing... please wait..."):
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            # Send image to Flask backend
            files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
            try:
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    mask_bytes = bytes(data["mask_bytes"])
                    mask_img = Image.open(io.BytesIO(mask_bytes))
                    st.image(mask_img, caption="🩺 Predicted Mask", use_column_width=True)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"❌ Could not connect to backend: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Flask")
