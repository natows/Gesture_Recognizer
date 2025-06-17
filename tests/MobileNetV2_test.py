import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle, time, pyautogui

model = load_model('../models/MobileNetV2/mobilenetv2_extended_model.keras')
with open('../models/MobileNetV2/extended_label_encoder_mobilenet.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

last_gesture = None
last_action_time = 0
action_cooldown = 1.0

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
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                
                hand_img = tf.keras.applications.mobilenet_v2.preprocess_input(hand_img)
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img, verbose=0)  # verbose=0 wyłącza logi
                pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                confidence = np.max(prediction)

                text = f"{pred_label}: {confidence:.2f}"
                cv2.putText(frame, text, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # current_time = time.time()
                # if pred_label == "V" and (pred_label != last_gesture or current_time - last_action_time > action_cooldown):
                #     pyautogui.scroll(-500)
                #     last_action_time = current_time
                # elif pred_label == "l" and (pred_label != last_gesture or current_time - last_action_time > action_cooldown):
                #     pyautogui.scroll(500)
                #     last_action_time = current_time
                # elif pred_label == "ok" and (pred_label != last_gesture or current_time - last_action_time > action_cooldown):
                #     pyautogui.hotkey('ctrl', 'w')
                #     last_action_time = current_time
                # elif pred_label == "palm" and (pred_label != last_gesture or current_time - last_action_time > action_cooldown):
                #     pyautogui.hotkey('alt', 'left')
                #     last_action_time = current_time
                # elif pred_label == "fist" and (pred_label != last_gesture or current_time - last_action_time > action_cooldown):
                #     pyautogui.press('space')
                #     last_action_time = current_time
                # elif pred_label == "thumb" and (pred_label != last_gesture or current_time - last_action_time > action_cooldown):
                #     pyautogui.press('volumeup')
                #     last_action_time = current_time

                # last_gesture = pred_label  


                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MobileNetV2 Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()