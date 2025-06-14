import cv2
import mediapipe as mp
import os
import csv
import time

SEQUENCE_LENGTH = 10
CSV_FILE = 'data_sources/sequences.csv'
LABELS = {
    ord('1'): 'finger_up',
    ord('2'): 'finger_down', 
    ord('3'): 'zoom_in',
    ord('4'): 'zoom_out'
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tworzenie pliku CSV z nagłówkami
if not os.path.exists('data_sources'):
    os.makedirs('data_sources')

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{coord}{i}_{t}' for t in range(SEQUENCE_LENGTH) for coord in ('x', 'y') for i in range(21)] + ['label']
        writer.writerow(header)

cap = cv2.VideoCapture(0)

# Sprawdź czy kamera działa
if not cap.isOpened():
    print("❌ Błąd: Nie można otworzyć kamery!")
    exit()

print("🎬 SEQUENCE LANDMARK COLLECTOR")
print("=" * 50)
print("STEROWANIE:")
print("1 - finger_up    2 - finger_down")
print("3 - zoom_in      4 - zoom_out") 
print("Q - zakończ")
print("=" * 50)

recording = False
current_label = None
sequence = []
sequence_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Błąd odczytu z kamery")
        break
    
    # ZAWSZE przetwarzaj obraz MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Wyświetl informacje na ekranie
    status_text = "🔴 NAGRYWANIE" if recording else "⏸️ GOTOWY"
    status_color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    if current_label:
        cv2.putText(frame, f"Gest: {current_label}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    if recording:
        progress = f"Klatki: {len(sequence)}/{SEQUENCE_LENGTH}"
        cv2.putText(frame, progress, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Instrukcje
    cv2.putText(frame, "1-4: wybierz gest, Q: wyjscie", (10, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Rysuj landmarks jeśli wykryto dłoń
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Jeśli nagrywamy sekwencję
            if recording:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                sequence.append(x_coords + y_coords)
                
                print(f"📹 Klatka {len(sequence)}/{SEQUENCE_LENGTH} - {current_label}")
                
                # Jeśli sekwencja pełna
                if len(sequence) >= SEQUENCE_LENGTH:
                    # Zapisz do CSV
                    flat_sequence = [coord for frame_data in sequence for coord in frame_data]
                    flat_sequence.append(current_label)
                    
                    with open(CSV_FILE, mode='a', newline='') as f:
                        csv.writer(f).writerow(flat_sequence)
                    
                    sequence_count += 1
                    print(f"✅ Zapisano sekwencję {sequence_count} dla '{current_label}'")
                    
                    # Reset
                    recording = False
                    sequence = []
                    current_label = None
                
                break  # Tylko jedna dłoń
    else:
        if recording:
            print("⚠️ Dłoń niewidoczna - kontynuuj gest...")
    
    # ZAWSZE wyświetl obraz
    cv2.imshow("Sequence Recording", frame)
    
    # Obsługa klawiszy
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key in LABELS and not recording:
        current_label = LABELS[key]
        recording = True
        sequence = []
        print(f"🔴 Rozpoczynam nagrywanie '{current_label}' - wykonuj gest!")

print(f"🎉 Zebrano {sequence_count} sekwencji!")
cap.release()
cv2.destroyAllWindows()