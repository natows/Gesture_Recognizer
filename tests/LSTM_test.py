import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import pyautogui
import time
from collections import deque
import json

# Wczytaj model i konfigurację
print("🤖 Ładowanie modelu LSTM...")
model = load_model('../models/LSTM/lstm_gesture_model.keras')

with open('../models/LSTM/label_encoder_lstm.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Wczytaj informacje o modelu
with open('../models/LSTM/model_info.json', 'r') as f:
    model_info = json.load(f)

SEQUENCE_LENGTH = model_info['sequence_length']
N_FEATURES_PER_FRAME = model_info['n_features_per_frame']

print(f"✅ Model wczytany - {model_info['classes']}")
print(f"📊 Długość sekwencji: {SEQUENCE_LENGTH} klatek")
print(f"🎯 Test accuracy modelu: {model_info['test_accuracy']:.2%}")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Bufor na sekwencję landmarks
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Stabilizacja predykcji
prediction_buffer = deque(maxlen=3)
last_gesture = None
last_action_time = 0
action_cooldown = 1.0  # Sekunda przerwy między akcjami

def extract_landmarks(hand_landmarks):
    """Ekstraktuje landmarks jako wektor [x0,y0,x1,y1,...]"""
    if hand_landmarks is None:
        return np.zeros(N_FEATURES_PER_FRAME)
    
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y])
    
    return np.array(landmarks)

def get_stable_prediction(gesture, confidence, threshold=0.7):
    """Stabilizuje predykcje używając bufora"""
    if confidence > threshold:
        prediction_buffer.append(gesture)
    
    if len(prediction_buffer) >= 2:
        predictions = list(prediction_buffer)
        # Znajdź najczęściej występujący gest
        most_common = max(set(predictions), key=predictions.count)
        if predictions.count(most_common) >= 2:
            return most_common
    
    return gesture if confidence > threshold else None

def perform_action(gesture):
    """Wykonuje akcję na podstawie rozpoznanego gestu"""
    global last_gesture, last_action_time
    
    current_time = time.time()
    
    
    # Sprawdź cooldown
    if gesture == last_gesture and current_time - last_action_time < action_cooldown:
        return
    
    if gesture == "finger_up":
        pyautogui.scroll(300)  # Scroll up
        print(f"📜 Scroll UP - {gesture}")
        
    elif gesture == "finger_down":
        pyautogui.scroll(-300)  # Scroll down
        print(f"📜 Scroll DOWN - {gesture}")
        
    elif gesture == "zoom_in":
        pyautogui.hotkey('ctrl', '+')  # Zoom in
        print(f"🔍 Zoom IN - {gesture}")
        
    elif gesture == "zoom_out":
        pyautogui.hotkey('ctrl', '-')  # Zoom out
        print(f"🔍 Zoom OUT - {gesture}")
    
    last_gesture = gesture
    last_action_time = current_time

# Główna pętla
cap = cv2.VideoCapture(0)

print("\n🎬 LSTM GESTURE RECOGNITION - TEST")
print("=" * 50)
print("KONTROLA GESTAMI:")
print("👆 finger_up   - Scroll UP")
print("👇 finger_down - Scroll DOWN") 
print("🔍 zoom_in     - Zoom IN (Ctrl++)")
print("🔍 zoom_out    - Zoom OUT (Ctrl+-)")
print("Q - wyjście")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Przetwarzanie MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    # Status bufora sekwencji
    buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
    cv2.putText(frame, buffer_status, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Rysuj landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Rysuj bounding box (dla wizualizacji)
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            xmin = max(int(min(x_coords) * w) - 20, 0)
            xmax = min(int(max(x_coords) * w) + 20, w)
            ymin = max(int(min(y_coords) * h) - 20, 0)
            ymax = min(int(max(y_coords) * h) + 20, h)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Ekstraktuj landmarks i dodaj do bufora
            landmarks = extract_landmarks(hand_landmarks)
            sequence_buffer.append(landmarks)
            
            # Jeśli mamy pełną sekwencję, wykonaj predykcję
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Przygotuj dane do predykcji
                sequence = np.array(list(sequence_buffer))
                sequence = np.expand_dims(sequence, axis=0)  # Dodaj batch dimension
                
                # Predykcja
                prediction = model.predict(sequence, verbose=0)[0]
                pred_class = np.argmax(prediction)
                confidence = np.max(prediction)
                gesture = label_encoder.inverse_transform([pred_class])[0]
                
                # Stabilizuj predykcję
                stable_gesture = get_stable_prediction(gesture, confidence)
                
                # Wyświetl wyniki
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                text = f"{gesture}: {confidence:.2f}"
                cv2.putText(frame, text, (xmin, ymin - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Wyświetl stabilny gest jeśli dostępny
                if stable_gesture:
                    cv2.putText(frame, f"Action: {stable_gesture}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Wykonaj akcję
                    perform_action(stable_gesture)
                
                # Wyświetl confidence bar
                bar_width = int(confidence * 200)
                cv2.rectangle(frame, (10, 100), (10 + bar_width, 120), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 100), (210, 120), (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            break  # Tylko jedna dłoń
    else:
        # Jeśli brak dłoni, dodaj zerowe landmarks
        zero_landmarks = np.zeros(N_FEATURES_PER_FRAME)
        sequence_buffer.append(zero_landmarks)
        
        cv2.putText(frame, "No hand detected", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Wyświetl ostatnią akcję
    current_time = time.time()
    if last_gesture and current_time - last_action_time < 2.0:
        action_text = f"Last action: {last_gesture}"
        cv2.putText(frame, action_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Instrukcje
    cv2.putText(frame, "Q - quit, Perform gestures for 1 second", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow("LSTM Gesture Recognition Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n🎉 Test zakończony!")
print("Dziękuję za używanie LSTM Gesture Recognition!")