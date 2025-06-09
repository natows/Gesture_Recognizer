import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle
import matplotlib.pyplot as plt


def load_leap_gesture_data(data_path='../data_sources/leapGestRecog'):
    images = []
    labels = []
    
    gesture_mapping = {
        '01_palm': 'palm',
        '02_l': 'l',
        '03_fist': 'fist',
        '04_fist_moved': 'fist_moved',
        '05_thumb': 'thumb',
        '06_index': 'index',
        '07_ok': 'ok',
        '08_palm_moved': 'palm_moved',
        '09_c': 'c',
        '10_down': 'down'
    }
    
    print("Wczytywanie danych z leapGestRecog...")

    for person_folder in sorted(os.listdir(data_path)):
        person_path = os.path.join(data_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
            
        print(f"Przetwarzanie osoby: {person_folder}")
        

        for gesture_folder in sorted(os.listdir(person_path)):
            gesture_path = os.path.join(person_path, gesture_folder)
            
            if not os.path.isdir(gesture_path):
                continue
                

            gesture_name = gesture_mapping.get(gesture_folder, gesture_folder)
            

            image_count = 0
            for image_file in os.listdir(gesture_path):
                if image_file.endswith(('.ppm', '.jpg', '.png', '.bmp')):
                    image_path = os.path.join(gesture_path, image_file)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = cv2.resize(image, (64, 64))
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = image.astype('float32') / 255.0
                            
                            images.append(image)
                            labels.append(gesture_name)
                            image_count += 1
                            
                    except Exception as e:
                        print(f"Błąd wczytywania {image_path}: {e}")
            
            print(f"{gesture_folder} ({gesture_name}): {image_count} obrazów")
    
    print(f"\nŁącznie wczytano: {len(images)} obrazów, {len(set(labels))} klas")
    print(f"Klasy: {sorted(set(labels))}")
    
    return np.array(images), np.array(labels)


    
    
def create_cnn_model(input_shape, num_classes): 
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape= input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"), 
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),  
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5)
    ])
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_cnn_model(images, labels, epochs, batch_size):
    input_shape = images.shape[1:] 
    num_classes = len(set(labels))  
    
    model = create_cnn_model(input_shape, num_classes)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    history = model.fit(images, labels_encoded, epochs=epochs, batch_size=batch_size,
            validation_split=0.2, shuffle=True)    
    return model, label_encoder, history


if __name__ == "__main__":
    images, labels = load_leap_gesture_data()

    print(f"Obrazy: {images.shape}, Etykiety: {labels.shape}")
    
    model, label_encoder, history = train_cnn_model(images, labels, epochs=20, batch_size=32)

    model.save('../models/CNN/cnn_model2.keras')
    with open('../models/CNN/label_encoder2.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Model i etykiety zostały zapisane.")

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Trening')
    plt.plot(history.history['val_accuracy'], label='Walidacja')
    plt.title('Dokładność')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy')
    plt.legend()

    # Strata
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Trening')
    plt.plot(history.history['val_loss'], label='Walidacja')
    plt.title('Strata')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show() 