import cv2
import mediapipe as mp
from utils.gesture_utils import get_landmark_list, count_fingers, recognize_gesture

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(rgb)

    gesture = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Get landmark coordinates
            lm_list = get_landmark_list(hand_landmarks, img.shape)

            # Count fingers
            finger_count = count_fingers(lm_list)

            # Recognize gesture
            gesture = recognize_gesture(finger_count)

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display gesture text
    cv2.putText(
        img,
        gesture,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    # Show webcam window
    cv2.imshow("Hand Gesture Recognition", img)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()