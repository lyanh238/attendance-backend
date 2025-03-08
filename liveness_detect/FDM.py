import numpy as np
import cv2
import keras
import os

model_path = os.path.join(os.path.dirname(__file__), "model_1.h5")

# Load the model
model = keras.models.load_model(model_path)


# Hàm dự đoán ảnh
def predict_image(image):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        print("Warning: Invalid image received in predict_image()")
        return None  # Trả về None để tránh lỗi

    # Resize ảnh
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Dự đoán
    prediction = model.predict(image)
    return "Fake" if prediction < 0.5 else "Real"
