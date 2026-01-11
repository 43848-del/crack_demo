import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Crack Detection",
    layout="centered"
)

st.title("📸 Crack Detection (Mobile Ready)")
st.write("ถ่ายรูปหรืออัปโหลดรูปเพื่อตรวจหารอยร้าว")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

uploaded_file = st.camera_input("📷 ถ่ายรูป")

if uploaded_file is None:
    uploaded_file = st.file_uploader(
        "หรืออัปโหลดรูปภาพ",
        type=["jpg", "png", "jpeg"]
    )

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("📊 ผลการตรวจจับ")
    if prediction > 0.5:
        st.error(f"⚠️ พบรอยร้าว ({prediction:.2f})")
    else:
        st.success(f"✅ ไม่พบรอยร้าว ({prediction:.2f})")
