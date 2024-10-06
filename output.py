import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

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

                img = cv2.resize(img, (64, 64))  # Thay đổi kích thước hình ảnh
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
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
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

# Huấn luyện mô hình
def train_model(data_dir):
    images, labels = load_dataset(data_dir)
    images = images.astype('float32') / 255.0  # Chuẩn hóa hình ảnh
    labels, unique_labels = convert_labels(labels)

    # Tạo mô hình
    model = create_model(len(unique_labels))

    # Huấn luyện mô hình
    model.fit(images, labels, epochs=10, batch_size=32)

    # Lưu mô hình với định dạng .keras
    model.save('gesture_recognition_model.keras')
    print("Mô hình đã được lưu.")
    
    return unique_labels  # Trả về nhãn duy nhất để sử dụng sau này

# Hàm nhận diện ký hiệu
def recognize_gestures(model, unique_labels):
    cap = cv2.VideoCapture(0)
    print("Bắt đầu nhận diện ký hiệu. Nhấn 'q' để dừng.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy hình ảnh từ camera.")
            break

        img = cv2.resize(frame, (64, 64))  # Thay đổi kích thước hình ảnh
        img = np.expand_dims(img, axis=0)  # Thêm chiều cho batch
        img = img.astype('float32') / 255.0  # Chuẩn hóa hình ảnh

        # Dự đoán ký hiệu
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        gesture_name = list(unique_labels.keys())[predicted_class]  # Lấy tên ký hiệu tương ứng

        # Hiển thị ký hiệu
        cv2.putText(frame, f'Ky Hieu: {gesture_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị khung hình
        cv2.imshow('Gesture Recognition', frame)

        # Nhấn 'q' để dừng nhận diện
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ví dụ sử dụng
if __name__ == "__main__":
    unique_labels = train_model('dataset')  # Huấn luyện và lấy nhãn
    # Tải lại mô hình đã lưu
    model = tf.keras.models.load_model('gesture_recognition_model.keras')
    recognize_gestures(model, unique_labels)  # Nhận diện ký hiệu
