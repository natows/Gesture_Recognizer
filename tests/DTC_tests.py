import cv2, pickle
import numpy as np
import pyautogui, time

cap = cv2.VideoCapture(0) 
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  
mp_draw = mp.solutions.drawing_utils

last_gesture = None
last_action_time = 0
action_cooldown = 2.0

 

with open('../models/DTC/DTC_extended.pkl', 'rb') as f:
    model = pickle.load(f)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        points = x_coords + y_coords
        

        points_np = np.array(points).reshape(1, -1)
        prediction = model.predict(points_np)[0]

        print(f"Predykcja: {prediction}")

        cv2.putText(frame, f'Gesture: {prediction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Live Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
