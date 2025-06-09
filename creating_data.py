import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

csv_file = 'data_sources/gestures.csv'

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
        writer.writerow(header)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection", img)
    key = cv2.waitKey(1) & 0xFF

    label = None
    if key == ord('1'):
        label = 'palm'
    elif key == ord('2'):
        label = 'I'
    elif key == ord('3'):
        label = "V"
    elif key == ord('4'):
        label = 'ok'
    elif key == ord('q'):
        break
    if label:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                row = x_coords + y_coords + [label]
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"[✓] Zapisano gest '{label}'")
        else:
            print("❗ Nie wykryto dłoni, nie zapisano.")

cap.release()
cv2.destroyAllWindows()
