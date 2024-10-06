import cv2
import os

# Hàm để tạo thư mục nếu nó không tồn tại
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Hàm để chụp hình ảnh
def capture_images(gesture_name):
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)

    create_directory('dataset')
    gesture_folder = os.path.join('dataset', gesture_name)
    create_directory(gesture_folder)

    print(f"Bắt đầu chụp hình cho ký hiệu '{gesture_name}'. Nhấn 's' để lưu hình ảnh, 'q' để dừng.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy hình ảnh từ camera.")
            break

        # Hiển thị khung hình
        cv2.imshow('Camera', frame)

        # Nhấn 's' để lưu hình ảnh
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img_name = f"{gesture_name}_{count + 1}.jpg"
            cv2.imwrite(os.path.join(gesture_folder, img_name), frame)
            print(f"Lưu hình ảnh: {img_name}")
            count += 1

        # Nhấn 'q' để dừng
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera
    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất việc chụp hình.")

# Ví dụ sử dụng
if __name__ == "__main__":
    gesture_name = input("Nhập tên ký hiệu bạn muốn chụp: ")
    capture_images(gesture_name)
