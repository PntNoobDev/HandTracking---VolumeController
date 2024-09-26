import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define finger tip landmarks (Mediapipe uses 21 landmarks)
# These are the landmarks of finger tips
tip_ids = [4, 8, 12, 16, 20]

def count_fingers(lm_list):
    # Initialize finger count
    fingers = []

    # Thumb (check for the x-axis movement)
    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:  # Thumb
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers (check for the y-axis)
    for id in range(1, 5):
        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0  # Variable to store total fingers count

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if lm_list:
                fingers_up = count_fingers(lm_list)
                total_fingers += fingers_up.count(1)  # Add the number of fingers up

            # Draw the hand landmarks on the image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the total finger count on the image
    cv2.putText(img, f'Total Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
