import cv2
import mediapipe as mp

# Khởi tạo các đối tượng Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Hàm để nhận diện cử chỉ tay và ánh xạ chúng với ký hiệu
def recognize_gesture(landmarks):
    # Ví dụ đơn giản để nhận diện một số cử chỉ tay cơ bản
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Ví dụ ánh xạ cử chỉ tay với ký hiệu chữ cái
    # Bạn cần thay thế bằng các cử chỉ cụ thể của ngôn ngữ ký hiệu mà bạn muốn nhận diện
    if thumb_tip.y < index_tip.y and index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y:
        return "Fist"  # Ví dụ tay nắm chặt, có thể ánh xạ với ký hiệu cụ thể
    elif thumb_tip.y < index_tip.y and index_tip.y < middle_tip.y and middle_tip.y > ring_tip.y and pinky_tip.y > ring_tip.y:
        return "Open Hand"  # Ví dụ tay mở, có thể ánh xạ với ký hiệu cụ thể
    # Thêm các cử chỉ tay khác và ánh xạ chúng với ký hiệu tương ứng
    return "Unknown"

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Không thể mở camera.")
        break

    # Chuyển đổi hình ảnh thành định dạng RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Nhận diện các tay
    results = hands.process(image_rgb)

    # Hiển thị các dấu vân tay trên hình ảnh
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Nhận diện cử chỉ tay
            gesture = recognize_gesture(hand_landmarks)
            
            # Vẽ các dấu vân tay và hiển thị ký hiệu tương ứng
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hiển thị ký hiệu tương ứng
            h, w, _ = image.shape
            cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị hình ảnh với các dấu vân tay và ký hiệu
    cv2.imshow('Hand Gesture Recognition', image)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
