import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
from collections import Counter

with open('../models/DTC/DTC_extended.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

CORRECT_SEQUENCE = ["thumb", "fist", "V", "ok", "rock"]  
TOTAL_GESTURES = len(CORRECT_SEQUENCE)

gesture_buffer = []
buffer_size = 10
confidence_threshold = 6

cooldown_time = 2.0
last_action_time = 0
gesture_hold_time = 1.0
current_gesture_start = 0
last_stable_gesture = None

user_sequence = []  
current_step = 0

status_messages = []
max_status_messages = 4

def add_status_message(message):
    """Dodaj wiadomość do historii statusu"""
    status_messages.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    if len(status_messages) > max_status_messages:
        status_messages.pop(0)

def get_stable_gesture(predictions):
    """Zwróć najczęstszy gest z bufora"""
    if len(predictions) < buffer_size:
        return None
    
    gesture_counts = Counter(predictions)
    most_common_gesture, count = gesture_counts.most_common(1)[0]
    
    if count >= confidence_threshold:
        return most_common_gesture
    return None

def draw_progress_bar(frame, current, total, y_pos=120):
    """Rysuj pasek postępu - BEZ ujawniania sekwencji"""
    bar_width = 400
    bar_height = 20
    x_start = 50
    
    cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height), 
                  (50, 50, 50), -1)
    
    progress_width = int((current / total) * bar_width)
    cv2.rectangle(frame, (x_start, y_pos), (x_start + progress_width, y_pos + bar_height),
                  (0, 255, 255), -1)  # Żółty kolor - neutralny
    
    cv2.putText(frame, f"Gesture: {current}/{total}", (x_start, y_pos - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_countdown_timer(frame, remaining_time, y_pos=150):
    """Rysuj odliczanie do następnego gestu"""
    if remaining_time > 0:
        cv2.putText(frame, f"Next gesture in: {remaining_time:.1f}s", (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def draw_gesture_hold_progress(frame, hold_progress, y_pos=180):
    """Rysuj postęp trzymania gestu"""
    if hold_progress > 0:
        bar_width = 200
        bar_height = 15
        x_start = 50
        
        cv2.rectangle(frame, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height),
                      (50, 50, 50), -1)
        
        progress_width = int(hold_progress * bar_width)
        color = (0, 255, 255) if hold_progress < 1.0 else (0, 255, 0)
        cv2.rectangle(frame, (x_start, y_pos), (x_start + progress_width, y_pos + bar_height),
                      color, -1)
        
        cv2.putText(frame, f"Hold gesture: {hold_progress*100:.0f}%", (x_start, y_pos - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def verify_sequence(user_seq, correct_seq):
    """Sprawdź czy sekwencja użytkownika jest poprawna"""
    if len(user_seq) != len(correct_seq):
        return False
    
    for i in range(len(user_seq)):
        if user_seq[i] != correct_seq[i]:
            return False
    
    return True

def show_final_result(frame, is_correct, user_seq, correct_seq):
    """Pokaż wynik weryfikacji"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 150), (590, 400), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    if is_correct:
        cv2.putText(frame, "ACCESS GRANTED", (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "Welcome!", (250, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "ACCESS DENIED", (140, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "Incorrect sequence!", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, "Your sequence:", (70, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    user_text = " -> ".join(user_seq)
    cv2.putText(frame, user_text, (70, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.putText(frame, "Correct sequence:", (70, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    correct_text = " -> ".join(correct_seq)
    cv2.putText(frame, correct_text, (70, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(" SECURE GESTURE VERIFICATION")
print("=" * 40)
print(f" Perform {TOTAL_GESTURES} gestures in sequence")
print("  Hold each gesture for 1 second")
print(" Press 'q' to quit")
print("=" * 40)

add_status_message(" Secure verification started")
add_status_message(f"Perform gesture {current_step + 1}/{TOTAL_GESTURES}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    current_time = time.time()
    predicted_gesture = "No hand detected"
    
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        points = x_coords + y_coords
        points_np = np.array(points).reshape(1, -1)
        
        prediction = model.predict(points_np)[0]
        predicted_gesture = prediction
        
        gesture_buffer.append(prediction)
        if len(gesture_buffer) > buffer_size:
            gesture_buffer.pop(0)
        
        stable_gesture = get_stable_gesture(gesture_buffer)
        
        if stable_gesture and current_step < TOTAL_GESTURES:
            if current_time - last_action_time > cooldown_time:
                
                if last_stable_gesture != stable_gesture:
                    current_gesture_start = current_time
                    last_stable_gesture = stable_gesture
                    add_status_message(f"Detected '{stable_gesture}' - hold...")
                
                hold_time = current_time - current_gesture_start
                hold_progress = min(hold_time / gesture_hold_time, 1.0)
                
                draw_gesture_hold_progress(frame, hold_progress)
                
                if hold_time >= gesture_hold_time:
                    user_sequence.append(stable_gesture)
                    current_step += 1
                    last_action_time = current_time
                    last_stable_gesture = None
                    gesture_buffer.clear()
                    
                    add_status_message(f" Gesture {current_step}/{TOTAL_GESTURES} recorded")
                    
                    if current_step == TOTAL_GESTURES:
                        is_correct = verify_sequence(user_sequence, CORRECT_SEQUENCE)
                        
                        add_status_message("🔍 Verifying sequence...")
                        time.sleep(1)  
                        
                        for _ in range(150):  
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.flip(frame, 1)
                                show_final_result(frame, is_correct, user_sequence, CORRECT_SEQUENCE)
                                cv2.imshow('Secure Gesture Verification', frame)
                                cv2.waitKey(33)
                        
                        print(f"\n🔍 VERIFICATION RESULTS:")
                        print(f"User sequence:    {' -> '.join(user_sequence)}")
                        print(f"Correct sequence: {' -> '.join(CORRECT_SEQUENCE)}")
                        print(f"Result: {' ACCESS GRANTED' if is_correct else ' ACCESS DENIED'}")
                        
                        break
                    else:
                        add_status_message(f"Next: gesture {current_step + 1}/{TOTAL_GESTURES}")
            
            else:
                remaining_cooldown = cooldown_time - (current_time - last_action_time)
                draw_countdown_timer(frame, remaining_cooldown)
    
    else:
        gesture_buffer.clear()
        last_stable_gesture = None
    
    
    cv2.putText(frame, "SECURE GESTURE VERIFICATION", (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Detected: {predicted_gesture}", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if current_step < TOTAL_GESTURES:
        cv2.putText(frame, f"Perform any gesture ({current_step + 1}/{TOTAL_GESTURES})", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    draw_progress_bar(frame, current_step, TOTAL_GESTURES)
    
    for i, message in enumerate(status_messages):
        y_pos = 300 + i * 25
        cv2.putText(frame, message, (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, "Hold gestures for 1 sec | Sequence verified at end | Press 'q' to quit", 
                (50, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.imshow('Secure Gesture Verification', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"\n Verification interrupted by user")
        print(f"Partial sequence: {' -> '.join(user_sequence)}")
        break

cap.release()
cv2.destroyAllWindows()

print("\n🔒 Secure verification session ended")