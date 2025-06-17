import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
import tensorflow as tf

class DTCMobileNetEnsemble:
    def __init__(self):
        print("Wczytywanie modeli...")
        
        with open('../models/DTC/DTC_extended.pkl', 'rb') as f:
            self.dtc_model = pickle.load(f)
            
        self.mobilenet_model = load_model('../models/MobileNetV2/mobilenetv2_extended_model.keras')
        with open('../models/MobileNetV2/extended_label_encoder_mobilenet.pkl', 'rb') as f:
            self.mobilenet_label_encoder = pickle.load(f)

        print(f" MobileNet klasy: {self.mobilenet_label_encoder.classes_}")
        
        self.weights = {
            'dtc': 0.6,        
            'mobilenet': 0.4  
        }
        
        print(" Modele wczytane! (DTC + MobileNetV2)")
    
    def prepare_dtc_landmarks(self, hand_landmarks):
        """POPRAWNY FORMAT dla DTC - jak w gestures.csv"""
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]  # 21 punktów x
        y_coords = [lm.y for lm in hand_landmarks.landmark]  # 21 punktów y
        
        landmarks_dtc_format = x_coords + y_coords  # [x0,x1,...,x20,y0,y1,...,y20]
        
        print(f" DTC landmarks format: len={len(landmarks_dtc_format)}")
        print(f" First 5 X coords: {x_coords[:5]}")
        print(f" First 5 Y coords: {y_coords[:5]}")
        
        return np.array(landmarks_dtc_format)
    
    def predict_dtc(self, hand_landmarks):
        """Predykcja DTC z poprawnym formatem landmarks"""
        try:
            landmarks_dtc = self.prepare_dtc_landmarks(hand_landmarks)

            dtc_pred = self.dtc_model.predict([landmarks_dtc])[0]
            print(f"DTC prediction: {dtc_pred}")
            return dtc_pred
        except Exception as e:
            print(f"Błąd DTC: {e}")
            import traceback
            traceback.print_exc()
            return "error"
    
    def predict_mobilenet(self, image):
        """Predykcja MobileNet - używa label encodera"""
        try:
            image_resized = cv2.resize(image, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

            image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_rgb)
            image_batch = np.expand_dims(image_preprocessed, axis=0)

            mobilenet_probs = self.mobilenet_model.predict(image_batch, verbose=0)[0]
            pred_class = np.argmax(mobilenet_probs)

            mobilenet_pred = self.mobilenet_label_encoder.inverse_transform([pred_class])[0]
            confidence = np.max(mobilenet_probs)
            
            print(f"MobileNet prediction: {mobilenet_pred} ({confidence:.3f})")
            
            return {
                'prediction': mobilenet_pred,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Błąd MobileNet: {e}")
            return {'prediction': 'error', 'confidence': 0.0}
    
    def predict_ensemble(self, hand_landmarks, image):
        """Ensemble z poprawnym formatowaniem"""
        

        dtc_pred = self.predict_dtc(hand_landmarks)  
        mobilenet_result = self.predict_mobilenet(image)
        
        print(f" DTC: {dtc_pred}")
        print(f" MobileNet: {mobilenet_result['prediction']} ({mobilenet_result['confidence']:.3f})")
        
        if dtc_pred == mobilenet_result['prediction']:
            return {
                'prediction': dtc_pred,
                'confidence': 0.95,
                'consensus': 'AGREEMENT',
                'votes': [dtc_pred, mobilenet_result['prediction']]
            }
        
        elif mobilenet_result['confidence'] > 0.8:
            return {
                'prediction': mobilenet_result['prediction'],
                'confidence': mobilenet_result['confidence'],
                'consensus': 'MOBILENET_CONFIDENT',
                'votes': [dtc_pred, mobilenet_result['prediction']]
            }
        
        else:
            return {
                'prediction': dtc_pred,
                'confidence': 0.7,
                'consensus': 'DTC_FALLBACK',
                'votes': [dtc_pred, mobilenet_result['prediction']]
            }

def main():
    print(" DTC + MOBILENETV2 ENSEMBLE (LANDMARKS FIXED)")
    print("=" * 60)

    cap = cv2.VideoCapture(0) 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    ensemble_model = DTCMobileNetEnsemble()

    print(" Uruchamianie kamery... (naciśnij 'q' aby wyjść)")

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
                result_pred = ensemble_model.predict_ensemble(hand_landmarks, frame)
                
                gesture = result_pred['prediction']
                confidence = result_pred['confidence']
                consensus = result_pred['consensus']
                votes = result_pred['votes']
                
                print(f" FINAL: {gesture} | {confidence:.1%} | {consensus}")
                print(f"    Votes: {votes}")
                print("-" * 40)
                
                cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {confidence:.1%}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f'Method: {consensus}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, f"DTC: {votes[0]}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"MobileNet: {votes[1]}", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
            except Exception as e:
                print(f" Błąd ensemble: {e}")
                import traceback
                traceback.print_exc()

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('DTC + MobileNet Ensemble (Landmarks Fixed)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Test zakończony!")

if __name__ == "__main__":
    main()