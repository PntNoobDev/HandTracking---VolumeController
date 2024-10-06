import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hàm để tải dataset
def load_dataset(data_dir):
    images = []
    labels = []

    # Lặp qua từng thư mục trong dataset
    for label in os.listdir(data_dir):
        gesture_folder = os.path.join(data_dir, label)
        if os.path.isdir(gesture_folder):
            for img_name in os.listdir(gesture_folder):
                img_path = os.path.join(gesture_folder, img_name)
                img = cv2.imread(img_path)

                # Kiểm tra nếu hình ảnh không hợp lệ
                if img is None:
                    print(f"Không thể đọc hình ảnh: {img_path}")
                    continue

                img = cv2.resize(img, (128, 128))  # Thay đổi kích thước hình ảnh
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Chuyển đổi nhãn thành số
def convert_labels(labels):
    unique_labels = {label: idx for idx, label in enumerate(np.unique(labels))}
    return np.array([unique_labels[label] for label in labels]), unique_labels

# Tạo mô hình CNN
def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình với tăng cường dữ liệu
def train_model(data_dir):
    images, labels = load_dataset(data_dir)
    images = images.astype('float32') / 255.0  # Chuẩn hóa hình ảnh
    labels, unique_labels = convert_labels(labels)

    # Tăng cường dữ liệu
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(images)

    # Tạo mô hình
    model = create_model(len(unique_labels))

    # Huấn luyện mô hình
    model.fit(datagen.flow(images, labels, batch_size=32), epochs=20)  # Tăng số lượng epochs

    # Lưu mô hình với định dạng .keras
    model.save('gesture_recognition_model.keras')
    print("Mô hình đã được lưu.")

# Ví dụ sử dụng
if __name__ == "__main__":
    train_model('dataset')
