import streamlit as st
import numpy as np
from PIL import Image
import keras
import pandas as pd
from datetime import datetime
import os

@st.cache_resource
def load_model():
    return keras.models.load_model("model.keras")

st.title("📸 Crack Detection (Mobile Ready)")
st.write("ถ่ายรูปหรืออัปโหลดรูปเพื่อตรวจหารอยร้าว")

model = load_model()

# ===== Camera =====
camera_img = st.camera_input("📷 ถ่ายรูปด้วยกล้อง")

# ===== Upload =====
uploaded = st.file_uploader("หรืออัปโหลดรูป", type=["jpg", "png", "jpeg"])

img_source = camera_img if camera_img else uploaded

if img_source:
    img = Image.open(img_source).convert("RGB")
    img_resized = img.resize((256, 256))

    img_arr = np.array(img_resized) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = float(model.predict(img_arr)[0][0])

    st.image(img, caption="ภาพที่ใช้วิเคราะห์", use_column_width=True)
    st.write(f"🔍 Crack probability: **{pred:.3f}**")

    # ===== Save result =====
    result = {
        "timestamp": datetime.now().isoformat(),
        "probability": pred,
        "source": "camera" if camera_img else "upload"
    }

    df = pd.DataFrame([result])

    if not os.path.exists("results.csv"):
        df.to_csv("results.csv", index=False)
    else:
        df.to_csv("results.csv", mode="a", header=False, index=False)

    st.success("✅ บันทึกผลเรียบร้อย")
