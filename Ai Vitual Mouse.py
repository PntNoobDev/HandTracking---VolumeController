import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize Mediapipe and hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Screen size to scale the mouse movement
screen_width, screen_height = pyautogui.size()

# Variables to track previous state of fingers
index_finger_raised = False
middle_finger_raised = False

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally and convert color to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(frame_rgb)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks for index and middle fingers (tip and 2nd joint)
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_middle_joint = hand_landmarks.landmark[7]
            
            middle_finger_tip = hand_landmarks.landmark[12]
            middle_finger_middle_joint = hand_landmarks.landmark[11]

            # Scale the coordinates to the screen size for mouse movement
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Move the mouse cursor based on the index finger position
            pyautogui.moveTo(x, y)

            # Calculate distances to detect if fingers are straight
            distance_index_finger = calculate_distance(index_finger_tip, index_finger_middle_joint)
            distance_middle_finger = calculate_distance(middle_finger_tip, middle_finger_middle_joint)

            # Detect if the index and middle fingers are raised (not bent)
            is_index_finger_raised = distance_index_finger > 0.05  # Adjust this threshold as needed
            is_middle_finger_raised = distance_middle_finger > 0.05  # Adjust this threshold as needed

            # Left-click: When both index and middle fingers are raised
            if is_index_finger_raised and is_middle_finger_raised and not (index_finger_raised and middle_finger_raised):
                pyautogui.click()
            index_finger_raised = is_index_finger_raised  # Update the state
            middle_finger_raised = is_middle_finger_raised  # Update the state

            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("AI Virtual Mouse", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
