import cv2
import mediapipe as mp
import random
import math
import time

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Generate a random point on the screen
def generate_random_point(width, height):
    x = random.randint(0, width)
    y = random.randint(0, height)
    return (x, y)

# Start webcam capture
cap = cv2.VideoCapture(0)
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize random point and score
target_point = generate_random_point(screen_width, screen_height)
point_radius = 20  # Size of the target point
threshold_distance = 50  # Distance threshold for "touching" the point
score = 0  # Initialize score

# Initialize countdown timer (60 seconds)
start_time = time.time()
countdown_time = 60  # Total game time in seconds

while True:
    success, img = cap.read()
    if not success:
        break

    # Calculate remaining time
    elapsed_time = time.time() - start_time
    remaining_time = int(countdown_time - elapsed_time)

    if remaining_time <= 0:
        # Game over when time runs out
        cv2.putText(img, 'Game Over', (screen_width // 2 - 100, screen_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(img, f'Final Score: {score}', (screen_width // 2 - 100, screen_height // 2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow('Game', img)
        cv2.waitKey(3000)  # Display the final score for 3 seconds
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw the target point on the screen
    cv2.circle(img, target_point, point_radius, (0, 0, 255), -1)

    # Display score and remaining time
    cv2.putText(img, f'Score: {score}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Time: {remaining_time}s', (screen_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip position (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = img.shape
            index_finger_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

            # Draw landmarks and connections on the hand
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate the distance between the finger and the point
            distance = calculate_distance(index_finger_coords, target_point)

            # If the hand is close to the point, generate a new random point and increase score
            if distance < threshold_distance:
                target_point = generate_random_point(screen_width, screen_height)
                score += 1

    # Display the result
    cv2.imshow('Game', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
