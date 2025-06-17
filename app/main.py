import cv2, pickle
import numpy as np
import pyautogui, time
from tensorflow.keras.models import load_model
import tensorflow as tf

cap = cv2.VideoCapture(0) 
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  
mp_draw = mp.solutions.drawing_utils

last_gesture = None
last_action_time = 0
action_cooldown = 1.0

print("📋 Dostępne klasyfikatory:")
print("1. DTC")
print("2. MobileNetV2")
print("3. Ensemble (DTC + MobileNetV2)")

chosen_classifier = input("Wybierz klasyfikator (1/2/3): ").strip()

if chosen_classifier == "1":
    print("Ładowanie DTC...")
    with open('../models/DTC/DTC_extended.pkl', 'rb') as f:
        dtc_model = pickle.load(f)
    classifier_type = "DTC"
    
elif chosen_classifier == "2":
    print("Ładowanie MobileNetV2...")
    mobilenet_model = load_model('../models/MobileNetV2/mobilenetv2_extended_model.keras')
    with open('../models/MobileNetV2/extended_label_encoder_mobilenet.pkl', 'rb') as f:
        mobilenet_label_encoder = pickle.load(f)
    classifier_type = "MobileNetV2"
    
elif chosen_classifier == "3":
    print("Ładowanie Ensemble...")
    with open('../models/DTC/DTC_extended.pkl', 'rb') as f:
        dtc_model = pickle.load(f)
    mobilenet_model = load_model('../models/MobileNetV2/mobilenetv2_extended_model.keras')
    with open('../models/MobileNetV2/extended_label_encoder_mobilenet.pkl', 'rb') as f:
        mobilenet_label_encoder = pickle.load(f)
    classifier_type = "Ensemble"
    
else:
    print("Nieprawidłowy wybór!")
    exit()

print(f"Model {classifier_type} załadowany!")

ACTIONS = {
    "V": "scrolldown", 
    "l": "scrollup", 
    "L": "scrollup",   
    "ok": "volumedown",
    "palm": "space",
    "rock": "volumemute",
    "fist": "ctrl + w",
    "thumb": "volumeup",
}

def predict_dtc(hand_landmarks):
    """Predykcja DTC"""
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    points = x_coords + y_coords
    points_np = np.array(points).reshape(1, -1)
    return dtc_model.predict(points_np)[0]

def predict_mobilenet(frame):
    """Predykcja MobileNetV2"""
    image_resized = cv2.resize(frame, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_rgb)
    image_batch = np.expand_dims(image_preprocessed, axis=0)
    
    probs = mobilenet_model.predict(image_batch, verbose=0)[0]
    pred_class = np.argmax(probs)
    prediction = mobilenet_label_encoder.inverse_transform([pred_class])[0]
    confidence = np.max(probs)
    
    return prediction, confidence

def predict_ensemble(hand_landmarks, frame):
    """Predykcja Ensemble"""
    dtc_pred = predict_dtc(hand_landmarks)
    mobilenet_pred, mobilenet_conf = predict_mobilenet(frame)
    
    if dtc_pred == mobilenet_pred:
        return dtc_pred, 0.95  
    elif mobilenet_conf > 0.8:
        return mobilenet_pred, mobilenet_conf  
    else:
        return dtc_pred, 0.7  

print("📹 Uruchamianie kamery... (naciśnij 'q' aby wyjść)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        
        try:
            if classifier_type == "DTC":
                prediction = predict_dtc(hand_landmarks)
                confidence = 0.95
                
            elif classifier_type == "MobileNetV2":
                prediction, confidence = predict_mobilenet(frame)
                
            elif classifier_type == "Ensemble":
                prediction, confidence = predict_ensemble(hand_landmarks, frame)

            print(f" Predykcja: {prediction} ({confidence:.1%})")

            cv2.putText(frame, f'Gesture: {prediction}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {confidence:.1%}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Model: {classifier_type}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            current_time = time.time()
            if (prediction in ACTIONS and 
                confidence > 0.7 and  
                (prediction != last_gesture or current_time - last_action_time > action_cooldown)):
                
                action = ACTIONS[prediction]
                print(f" Executing action: {action}")
                
                if action == "scrolldown":
                    pyautogui.scroll(-500)  
                elif action == "scrollup":
                    pyautogui.scroll(500)
                elif action == "ctrl + w":
                    pyautogui.hotkey('ctrl', 'w')
                elif action == "volumedown":
                    pyautogui.press('volumedown')
                elif action == "volumeup":
                    pyautogui.press('volumeup')
                elif action == "volumemute":
                    pyautogui.press('volumemute')
                elif action == "space":
                    pyautogui.press('space')

                last_action_time = current_time

            last_gesture = prediction
            
        except Exception as e:
            print(f"Błąd predykcji: {e}")
            cv2.putText(frame, f'Error: {str(e)[:30]}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Live Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Aplikacja zakończona!")