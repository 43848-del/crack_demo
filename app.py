import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(
    page_title="Crack Detection Demo",
    layout="centered"
)

st.title("🧱 Crack Detection Demo")
st.write("อัปโหลดภาพเพื่อทดสอบการตรวจจับรอยร้าว")

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model("model.h5")

uploaded_file = st.file_uploader(
    "อัปโหลดภาพ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพต้นฉบับ", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("กำลังโหลดโมเดล..."):
        model = load_model()

    with st.spinner("กำลังวิเคราะห์..."):
        pred = model.predict(img)

    score = float(pred[0][0])

    if score > 0.5:
        st.error(f"⚠️ พบรอยร้าว (confidence {score:.2f})")
    else:
        st.success(f"✅ ไม่พบรอยร้าว (confidence {1-score:.2f})")