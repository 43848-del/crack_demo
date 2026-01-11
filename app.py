import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI Image Demo", layout="centered")

st.title("🧠 AI Image Analyzer (PyTorch)")
st.write("อัปโหลดรูป แล้วให้ AI วิเคราะห์")

# โหลดโมเดลตัวอย่าง (ยังไม่ train จริง)
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(224 * 224 * 3, 2),
    torch.nn.Softmax(dim=1)
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("📤 อัปโหลดรูป", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        confidence = torch.max(output).item()

    st.success(f"✅ วิเคราะห์เสร็จ (ความมั่นใจ: {confidence:.2f})")
