import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder


class PrintEpochMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        print(f"Epoka {epoch + 1}: acc = {acc:.4f}, val_acc = {val_acc:.4f}, loss = {loss:.4f}, val_loss = {val_loss:.4f}")


def load_custom_gesture_data(data_path='../dataset'):
    """Wczytuje własne dane gestów z folderu dataset/"""
    images = []
    labels = []
    
    print("Wczytywanie własnych danych z dataset...")
    
    if not os.path.exists(data_path):
        print(f"Folder {data_path} nie istnieje - pomijam własne dane")
        return np.array([]), np.array([])
    
    for gesture_folder in os.listdir(data_path):
        gesture_path = os.path.join(data_path, gesture_folder)
        
        if not os.path.isdir(gesture_path):
            continue
            
        print(f"Przetwarzanie gestu: {gesture_folder}")
        
        image_count = 0
        for image_file in os.listdir(gesture_path):
            if image_file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(gesture_path, image_file)
                
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Identyczne przetwarzanie jak w load_leap_gesture_data
                        image = cv2.resize(image, (64, 64))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = image.astype('float32') / 255.0
                        
                        images.append(image)
                        labels.append(gesture_folder)
                        image_count += 1
                        
                except Exception as e:
                    print(f"Błąd wczytywania {image_path}: {e}")
        
        print(f"{gesture_folder}: {image_count} obrazów")
    
    return np.array(images), np.array(labels)


def load_leap_gesture_data(data_path='../data_sources/leapGestRecog'):
    images = []
    labels = []
    
    gesture_mapping = {
        '01_palm': 'palm',
        '02_l': 'l',
        '03_fist': 'fist',
        '05_thumb': 'thumb',
        '06_index': 'index',
        '07_ok': 'ok',
        '08_palm_moved': 'palm_moved',
        '09_c': 'c',
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


def combine_datasets():
    """Łączy dane z obu źródeł"""
    print("=== ŁĄCZENIE DANYCH Z OBU ŹRÓDEŁ ===")
    
    # Wczytaj dane z leapGestRecog
    leap_images, leap_labels = load_leap_gesture_data()
    
    # Wczytaj własne dane
    custom_images, custom_labels = load_custom_gesture_data()
    
    # Połącz dane
    if len(custom_images) > 0 and len(leap_images) > 0:
        all_images = np.concatenate([leap_images, custom_images], axis=0)
        all_labels = np.concatenate([leap_labels, custom_labels], axis=0)
    elif len(leap_images) > 0:
        all_images = leap_images
        all_labels = leap_labels
    elif len(custom_images) > 0:
        all_images = custom_images
        all_labels = custom_labels
    else:
        print("❌ Brak danych do trenowania!")
        return np.array([]), np.array([])
    
    print(f"\n📊 PODSUMOWANIE:")
    print(f"   - Dane z leapGestRecog: {len(leap_images)} obrazów")
    print(f"   - Własne dane: {len(custom_images)} obrazów")
    print(f"   - ŁĄCZNIE: {len(all_images)} obrazów")
    
    # Sprawdź rozkład klas
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\n🏷️  Rozkład klas:")
    for label, count in zip(unique_labels, counts):
        print(f"   - {label}: {count} obrazów")
    
    return all_images, all_labels


    
    
def create_cnn_model(input_shape, num_classes): 
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),  
        MaxPooling2D((2, 2)),
        Dropout(0.25),         
        
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(), 
        MaxPooling2D((2, 2)),
        Dropout(0.25),        
        
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(), 
        MaxPooling2D((2, 2)),
        Dropout(0.4),         
        
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.6),         
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_cnn_model(images, labels, epochs, batch_size):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        PrintEpochMetrics()
    ]
    
    model = create_cnn_model(images.shape[1:], len(set(labels)))

    history = model.fit(
        datagen.flow(X_train, y_train_enc, batch_size=batch_size),
        validation_data=(X_val, y_val_enc),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, label_encoder, history


if __name__ == "__main__":
    # ZMIANA: Użyj combine_datasets() zamiast load_leap_gesture_data()
    images, labels = combine_datasets()
    
    if len(images) == 0:
        print("❌ Brak danych do trenowania!")
        exit()

    print(f"Obrazy: {images.shape}, Etykiety: {labels.shape}")
    
    model, label_encoder, history = train_cnn_model(images, labels, epochs=20, batch_size=32)

    model.save('../models/CNN/cnn_third_model.keras')
    with open('../models/CNN/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Model i etykiety zostały zapisane.")

    plt.figure(figsize=(12, 5))

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




# Epoch 1/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 128ms/step - accuracy: 0.1895 - loss: 2.7160Epoka 1: acc = 0.2396, val_acc = 0.1840, loss = 2.1937, val_loss = 4.9233
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 65s 136ms/step - accuracy: 0.1896 - loss: 2.7148 - val_accuracy: 0.1840 - val_loss: 4.9233 - learning_rate: 0.0010
# Epoch 2/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 133ms/step - accuracy: 0.3999 - loss: 1.6066Epoka 2: acc = 0.4736, val_acc = 0.8807, loss = 1.4095, val_loss = 0.4292
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 61s 139ms/step - accuracy: 0.4001 - loss: 1.6062 - val_accuracy: 0.8807 - val_loss: 0.4292 - learning_rate: 0.0010
# Epoch 3/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 131ms/step - accuracy: 0.6400 - loss: 0.9848Epoka 3: acc = 0.6831, val_acc = 0.8467, loss = 0.8761, val_loss = 0.4650
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 60s 136ms/step - accuracy: 0.6401 - loss: 0.9845 - val_accuracy: 0.8467 - val_loss: 0.4650 - learning_rate: 0.0010
# Epoch 4/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 99ms/step - accuracy: 0.7676 - loss: 0.6602Epoka 4: acc = 0.7879, val_acc = 0.9813, loss = 0.6047, val_loss = 0.0636
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 45s 102ms/step - accuracy: 0.7676 - loss: 0.6601 - val_accuracy: 0.9813 - val_loss: 0.0636 - learning_rate: 0.0010
# Epoch 5/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - accuracy: 0.8296 - loss: 0.4987Epoka 5: acc = 0.8396, val_acc = 0.9667, loss = 0.4770, val_loss = 0.0945
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 32s 73ms/step - accuracy: 0.8296 - loss: 0.4986 - val_accuracy: 0.9667 - val_loss: 0.0945 - learning_rate: 0.0010
# Epoch 6/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 116ms/step - accuracy: 0.8657 - loss: 0.3963Epoka 6: acc = 0.8706, val_acc = 0.9777, loss = 0.3878, val_loss = 0.0574
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 53s 122ms/step - accuracy: 0.8657 - loss: 0.3963 - val_accuracy: 0.9777 - val_loss: 0.0574 - learning_rate: 0.0010
# Epoch 7/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 128ms/step - accuracy: 0.8931 - loss: 0.3315Epoka 7: acc = 0.8963, val_acc = 0.9960, loss = 0.3179, val_loss = 0.0283
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 59s 133ms/step - accuracy: 0.8931 - loss: 0.3315 - val_accuracy: 0.9960 - val_loss: 0.0283 - learning_rate: 0.0010
# Epoch 8/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.9144 - loss: 0.2661Epoka 8: acc = 0.9142, val_acc = 0.9737, loss = 0.2715, val_loss = 0.0838
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 60s 137ms/step - accuracy: 0.9144 - loss: 0.2661 - val_accuracy: 0.9737 - val_loss: 0.0838 - learning_rate: 0.0010
# Epoch 9/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 128ms/step - accuracy: 0.9182 - loss: 0.2605Epoka 9: acc = 0.9187, val_acc = 0.9947, loss = 0.2619, val_loss = 0.0187
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 59s 134ms/step - accuracy: 0.9182 - loss: 0.2605 - val_accuracy: 0.9947 - val_loss: 0.0187 - learning_rate: 0.0010
# Epoch 10/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 98ms/step - accuracy: 0.9252 - loss: 0.2480Epoka 10: acc = 0.9316, val_acc = 0.9947, loss = 0.2262, val_loss = 0.0193
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 45s 102ms/step - accuracy: 0.9252 - loss: 0.2479 - val_accuracy: 0.9947 - val_loss: 0.0193 - learning_rate: 0.0010
# Epoch 11/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 117ms/step - accuracy: 0.9413 - loss: 0.1996Epoka 11: acc = 0.9438, val_acc = 0.9953, loss = 0.1896, val_loss = 0.0166
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 54s 123ms/step - accuracy: 0.9413 - loss: 0.1996 - val_accuracy: 0.9953 - val_loss: 0.0166 - learning_rate: 0.0010
# Epoch 12/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 126ms/step - accuracy: 0.9470 - loss: 0.1774Epoka 12: acc = 0.9432, val_acc = 0.9940, loss = 0.1884, val_loss = 0.0210
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 58s 132ms/step - accuracy: 0.9470 - loss: 0.1775 - val_accuracy: 0.9940 - val_loss: 0.0210 - learning_rate: 0.0010
# Epoch 13/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 126ms/step - accuracy: 0.9472 - loss: 0.1834Epoka 13: acc = 0.9451, val_acc = 0.9897, loss = 0.1890, val_loss = 0.0241
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 58s 132ms/step - accuracy: 0.9472 - loss: 0.1834 - val_accuracy: 0.9897 - val_loss: 0.0241 - learning_rate: 0.0010
# Epoch 14/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 128ms/step - accuracy: 0.9509 - loss: 0.1789Epoka 14: acc = 0.9514, val_acc = 0.9967, loss = 0.1685, val_loss = 0.0111
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 59s 134ms/step - accuracy: 0.9509 - loss: 0.1788 - val_accuracy: 0.9967 - val_loss: 0.0111 - learning_rate: 0.0010
# Epoch 15/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 130ms/step - accuracy: 0.9584 - loss: 0.1448Epoka 15: acc = 0.9566, val_acc = 0.9887, loss = 0.1521, val_loss = 0.0292
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 60s 136ms/step - accuracy: 0.9583 - loss: 0.1448 - val_accuracy: 0.9887 - val_loss: 0.0292 - learning_rate: 0.0010
# Epoch 16/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 125ms/step - accuracy: 0.9528 - loss: 0.1520Epoka 16: acc = 0.9537, val_acc = 0.9997, loss = 0.1510, val_loss = 0.0021
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 57s 131ms/step - accuracy: 0.9528 - loss: 0.1520 - val_accuracy: 0.9997 - val_loss: 0.0021 - learning_rate: 0.0010
# Epoch 17/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 132ms/step - accuracy: 0.9570 - loss: 0.1474Epoka 17: acc = 0.9575, val_acc = 0.9953, loss = 0.1499, val_loss = 0.0136
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 85s 138ms/step - accuracy: 0.9570 - loss: 0.1474 - val_accuracy: 0.9953 - val_loss: 0.0136 - learning_rate: 0.0010
# Epoch 18/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 126ms/step - accuracy: 0.9568 - loss: 0.1488Epoka 18: acc = 0.9595, val_acc = 0.9960, loss = 0.1452, val_loss = 0.0248
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 57s 129ms/step - accuracy: 0.9568 - loss: 0.1488 - val_accuracy: 0.9960 - val_loss: 0.0248 - learning_rate: 0.0010
# Epoch 19/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 119ms/step - accuracy: 0.9563 - loss: 0.1589Epoka 19: acc = 0.9612, val_acc = 0.9987, loss = 0.1392, val_loss = 0.0071
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 55s 124ms/step - accuracy: 0.9563 - loss: 0.1588 - val_accuracy: 0.9987 - val_loss: 0.0071 - learning_rate: 0.0010
# Epoch 20/20
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 0s 142ms/step - accuracy: 0.9676 - loss: 0.1125Epoka 20: acc = 0.9704, val_acc = 0.9963, loss = 0.1036, val_loss = 0.0099
# 438/438 ━━━━━━━━━━━━━━━━━━━━ 65s 148ms/step - accuracy: 0.9676 - loss: 0.1125 - val_accuracy: 0.9963 - val_loss: 0.0099 - learning_rate: 5.0000e-04
