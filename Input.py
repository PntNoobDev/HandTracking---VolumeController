import cv2
import os
import numpy as np

# Tạo thư mục để lưu dữ liệu ký hiệu tay
if not os.path.exists("gesture_data"):
    os.makedirs("gesture_data")

def capture_gesture(label):
    cap = cv2.VideoCapture(0)
    print("Press 'S' to save the gesture and 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị khung camera
        cv2.imshow('Gesture Capture', frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            # Lưu hình ảnh của ký hiệu tay
            file_name = f'gesture_data/{label}_{len(os.listdir("gesture_data"))}.jpg'
            cv2.imwrite(file_name, frame)
            print(f"Gesture saved as {file_name}")
        
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Cho người dùng nhập ký tự mà họ muốn gán cho ký hiệu tay
label = input("Nhập ký tự bạn muốn gán cho ký hiệu tay: ")
capture_gesture(label)
