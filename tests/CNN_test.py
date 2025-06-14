import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import pyautogui
import time


model = load_model('../models/CNN/cnn_custom2_only_model.keras')
with open('../models/CNN/label_encoder_custom2_only.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_coords) * w)
            xmax = int(max(x_coords) * w)
            ymin = int(min(y_coords) * h)
            ymax = int(max(y_coords) * h)

            margin = 20
            xmin = max(xmin - margin, 0)
            xmax = min(xmax + margin, w)
            ymin = max(ymin - margin, 0)
            ymax = min(ymax + margin, h)

            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size > 0:
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = hand_img.astype('float32') / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img)
                pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                cv2.putText(frame, pred_label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
