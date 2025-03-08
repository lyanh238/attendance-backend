import tensorflow as tf
from keras import layers, models
import cv2
import numpy as np
# Tạo mô hình CNN đơn giản
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification: real or fake
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình (cần dữ liệu huấn luyện)
# model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Dự đoán ảnh
def predict_image(image):
    image = cv2.resize(image, (128, 128))  # Resize ảnh về kích thước phù hợp
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    prediction = model.predict(image)
    return "Ảnh giả" if prediction > 0.5 else "Ảnh thật"

# Sử dụng hàm
image_path = "D:/BTL/face-reidentification/liveness_detect/dataset/test/fake/Fake (9).jpg"
image = cv2.imread(image_path)
print(predict_image(image))