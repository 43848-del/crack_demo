import tensorflow as tf

model = tf.keras.models.load_model(
    "model.keras",
    compile=False
)

model.save("model_tf.keras")
model.save("model_tf.h5")

print("✅ แปลงโมเดลสำเร็จ")
