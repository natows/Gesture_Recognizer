import cv2
import os
import mediapipe as mp

gesture_name = "fist"         
output_dir = f"./dataset/{gesture_name}"
os.makedirs(output_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)     
img_count = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

print(f"Zbieranie gestów '{gesture_name}':")
print("- Pokaż dłoń przed kamerą")
print("- 's' aby zapisać zdjęcie dłoni")
print("- 'q' aby zakończyć")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    hand_detected = False
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_detected = True
            
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            xmin = max(int(min(x_coords) * w) - 20, 0)
            xmax = min(int(max(x_coords) * w) + 20, w)
            ymin = max(int(min(y_coords) * h) - 20, 0)
            ymax = min(int(max(y_coords) * h) + 20, h)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_bbox = (xmin, ymin, xmax, ymax)

    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = f"Dłoń: {'WYKRYTA' if hand_detected else 'BRAK'}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"{gesture_name} - Zdjęć: {img_count}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Zbieranie gestów", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and hand_detected:
        xmin, ymin, xmax, ymax = hand_bbox
        hand_img = frame[ymin:ymax, xmin:xmax]
        
        if hand_img.size > 0:
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            
            img_path = os.path.join(output_dir, f"{gesture_name}_{img_count:04d}.jpg")
            cv2.imwrite(img_path, hand_img) 
            
            print(f"[✓] Zapisano: {img_path} (rozmiar: {hand_img.shape})")
            img_count += 1
        else:
            print("❌ Błąd wycinania dłoni")
            
    elif key == ord('s') and not hand_detected:
        print("❌ Nie wykryto dłoni - nie zapisano")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 Zebrano {img_count} zdjęć gestu '{gesture_name}'")
print("💡 Zmień 'gesture_name' i uruchom ponownie dla innych gestów")