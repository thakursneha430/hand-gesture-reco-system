import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("models/gesture_model.pkl","rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            landmarks = []

            for lm in handLms.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # use first 5 features for demo
            data = np.array(landmarks[:5]).reshape(1,-1)

            prediction = model.predict(data)

            cv2.putText(img, prediction[0], (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()