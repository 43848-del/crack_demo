import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Crack Detection Demo",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Crack Detection Demo")
st.write("ถ่ายรูปหรืออัปโหลดภาพ เพื่อทดสอบโมเดล")

# ===============================
# Load model (cache)
# ===============================
@st.cache_resource
def load_model():
    # เปลี่ยนชื่อไฟล์ให้ตรงกับของคุณ
    return tf.keras.models.load_model("model.keras")

try:
    model = load_model()
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error("❌ โหลดโมเดลไม่สำเร็จ")
    st.exception(e)
    st.stop()

# ===============================
# Image preprocessing
# ===============================
IMG_SIZE = 224  # ต้องตรงกับตอน train

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# Input method
# ===============================
tab1, tab2 = st.tabs(["📷 ถ่ายรูป", "📁 อัปโหลดไฟล์"])

image = None

with tab1:
    camera_image = st.camera_input("ถ่ายรูปด้วยกล้อง")
    if camera_image is not None:
        image = Image.open(camera_image)

with tab2:
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปภาพ",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# ===============================
# Prediction
# ===============================
if image is not None:
    st.image(image, caption="ภาพที่ใช้ทดสอบ", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        with st.spinner("กำลังประมวลผล..."):
            img_tensor = preprocess_image(image)
            prediction = model.predict(img_tensor)[0]

        # ===== ปรับตามโมเดลของคุณ =====
        if prediction.shape[0] == 1:
            # Binary classification (sigmoid)
            prob = float(prediction[0])
            label = "Crack" if prob > 0.5 else "No Crack"
            st.metric("ผลลัพธ์", label)
            st.progress(prob if prob > 0.5 else 1 - prob)

        else:
            # Multi-class (softmax)
            class_names = ["No Crack", "Crack"]  # แก้ชื่อได้
            class_id = int(np.argmax(prediction))
            confidence = float(prediction[class_id])

            st.metric("ผลลัพธ์", class_names[class_id])
            st.write(f"ความมั่นใจ: **{confidence:.2%}**")

            st.bar_chart(
                {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
            )

# ===============================
# Footer
# ===============================
st.divider()
st.caption(
    "🔗 แชร์ลิงก์หน้านี้ให้ผู้อื่นทดลองได้ทันที (Streamlit Cloud)"
)
