# distutils: language = c++
import tensorflow as tf
from keras import layers, models
from keras._tf_keras.keras.utils import image_dataset_from_directory
import os

# Khai báo kiểu biến cho Cython
BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
TRAIN_DIR: str = os.path.join(BASE_DIR, "train")
TEST_DIR: str = os.path.join(BASE_DIR, "test")
MODEL_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "liveness_plus.h5"))

# Kiểm tra thư mục dữ liệu
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Thư mục train hoặc test không tồn tại. Hãy kiểm tra lại đường dẫn.")

# Load dữ liệu với kiểm tra lỗi
try:
    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(128, 128),
        batch_size=32,
        label_mode="binary",
        class_names=["real", "fake"]
    )

    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=(128, 128),
        batch_size=32,
        label_mode="binary",
        class_names=["real", "fake"]
    )
except Exception as e:
    raise RuntimeError(f"Lỗi khi load dataset: {e}")

# Chuẩn hóa dữ liệu
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Xây dựng mô hình CNN
def train_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
    history = model.fit(train_ds, epochs=20, validation_data=test_ds, verbose=1)

    # Lưu mô hình sau khi train
    model.save(MODEL_PATH)
    print(f"Mô hình đã được lưu tại: {MODEL_PATH}")

    return model
