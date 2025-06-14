import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class PrintEpochMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        print(f"Epoka {epoch + 1}: acc = {acc:.4f}, val_acc = {val_acc:.4f}, loss = {loss:.4f}, val_loss = {val_loss:.4f}")


def load_custom_gesture_data(data_path='../../dataset'):
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


def load_leap_gesture_data_by_person(data_path='../data_sources/leapGestRecog'):
    """Wczytuje dane grupując po osobach dla prawidłowego podziału"""
    person_data = {}  
    
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
    
    print("Wczytywanie danych z leapGestRecog (grupowanie po osobach)...")

    for person_folder in sorted(os.listdir(data_path)):
        person_path = os.path.join(data_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
            
        print(f"Przetwarzanie osoby: {person_folder}")
        person_data[person_folder] = {'images': [], 'labels': []}

        for gesture_folder in sorted(os.listdir(person_path)):
            gesture_path = os.path.join(person_path, gesture_folder)
            
            if not os.path.isdir(gesture_path):
                continue
                
            gesture_name = gesture_mapping.get(gesture_folder, gesture_folder)

            for image_file in os.listdir(gesture_path):
                if image_file.endswith(('.ppm', '.jpg', '.png', '.bmp')):
                    image_path = os.path.join(gesture_path, image_file)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = cv2.resize(image, (64, 64))
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = image.astype('float32') / 255.0
                            
                            person_data[person_folder]['images'].append(image)
                            person_data[person_folder]['labels'].append(gesture_name)
                            
                    except Exception as e:
                        print(f"Błąd wczytywania {image_path}: {e}")
    
    return person_data

def split_data_by_person(person_data, test_size=0.3, val_size=0.2):
    """Dzieli dane po osobach, nie po obrazkach"""
    from sklearn.model_selection import train_test_split
    
    person_ids = list(person_data.keys())
    print(f"Liczba osób w datasecie: {len(person_ids)}")
    
    train_persons, temp_persons = train_test_split(
        person_ids, test_size=test_size, random_state=42
    )

    val_persons, test_persons = train_test_split(
        temp_persons, test_size=0.5, random_state=42
    )
    
    print(f" Podział osób:")
    print(f"- Train: {len(train_persons)} osób")
    print(f"- Validation: {len(val_persons)} osób") 
    print(f"- Test: {len(test_persons)} osób")
    
    def collect_images_from_persons(persons):
        images, labels = [], []
        for person in persons:
            images.extend(person_data[person]['images'])
            labels.extend(person_data[person]['labels'])
        return np.array(images), np.array(labels)
    
    X_train, y_train = collect_images_from_persons(train_persons)
    X_val, y_val = collect_images_from_persons(val_persons)
    X_test, y_test = collect_images_from_persons(test_persons)
    
    print(f"Podział obrazów:")
    print(f"   - Train: {len(X_train)} obrazów")
    print(f"   - Validation: {len(X_val)} obrazów")
    print(f"   - Test: {len(X_test)} obrazów")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
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



def load_leap_gesture_data(data_path='../data_sources/leapGestRecog'):
    """Wczytuje dane z Kaggle (stara funkcja - przywróć ją!)"""
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
            print(f"  - {gesture_folder} ({gesture_name})")

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
            
            print(f"    Wczytano: {image_count} obrazów")

    print(f"\nŁącznie z leapGestRecog: {len(images)} obrazów")
    return np.array(images), np.array(labels)


def combine_datasets_with_person_split():
    """Hybrydowe łączenie: Kaggle (podział po osobach) + Własne dane (1 osoba)"""
    print("=== HYBRYDOWE ŁĄCZENIE DANYCH ===")

    print("\n DANE Z KAGGLE (podział po osobach):")
    kaggle_person_data = load_leap_gesture_data_by_person()
    
    kaggle_train, kaggle_val, kaggle_test, y_kaggle_train, y_kaggle_val, y_kaggle_test = split_data_by_person(kaggle_person_data)
    
    print("\nWŁASNE DANE:")
    custom_images, custom_labels = load_custom_gesture_data()
    
    if len(custom_images) == 0:
        print("Brak własnych danych - używam tylko Kaggle")
        return kaggle_train, kaggle_val, kaggle_test, y_kaggle_train, y_kaggle_val, y_kaggle_test
    
    print(f"Wczytano {len(custom_images)} własnych obrazów")
    
    from sklearn.model_selection import train_test_split

    custom_train, custom_temp, y_custom_train, y_custom_temp = train_test_split(
        custom_images, custom_labels, test_size=0.2, random_state=42, stratify=custom_labels
    )

    custom_val, custom_test, y_custom_val, y_custom_test = train_test_split(
        custom_temp, y_custom_temp, test_size=0.5, random_state=42, stratify=y_custom_temp
    )
    
    print(f"Podział własnych danych:")
    print(f" - Train: {len(custom_train)} obrazów")
    print(f" - Val: {len(custom_val)} obrazów")
    print(f" - Test: {len(custom_test)} obrazów")

    print("\n ŁĄCZENIE DANYCH:")
    
    X_train = np.concatenate([kaggle_train, custom_train], axis=0)
    X_val = np.concatenate([kaggle_val, custom_val], axis=0)
    X_test = np.concatenate([kaggle_test, custom_test], axis=0)
    
    y_train = np.concatenate([y_kaggle_train, y_custom_train], axis=0)
    y_val = np.concatenate([y_kaggle_val, y_custom_val], axis=0)
    y_test = np.concatenate([y_kaggle_test, y_custom_test], axis=0)
    
    print(f"FINALNE DANE:")
    print(f" - Train: {len(X_train)} obrazów ({len(kaggle_train)} Kaggle + {len(custom_train)} własne)")
    print(f"   - Val: {len(X_val)} obrazów ({len(kaggle_val)} Kaggle + {len(custom_val)} własne)")
    print(f"   - Test: {len(X_test)} obrazów ({len(kaggle_test)} Kaggle + {len(custom_test)} własne)")

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"\nRozkład klas w treningu:")
    for label, count in zip(unique_train, counts_train):
        print(f"   - {label}: {count} obrazów")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_cnn_model_hybrid():
    """Trenuj model z hybrydowymi danymi"""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    X_train, X_val, X_test, y_train, y_val, y_test = combine_datasets_with_person_split()
    
    if len(X_train) == 0:
        print("Brak danych do trenowania!")
        return None

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)  

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        PrintEpochMetrics()
    ]

    model = create_cnn_model(X_train.shape[1:], len(set(y_train)))

    history = model.fit(
        datagen.flow(X_train, y_train_enc, batch_size=32),
        validation_data=(X_val, y_val_enc),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*60)
    print("FINALNA EWALUACJA MODELU (KAGGLE + WŁASNE DANE)")
    print("="*60)

    test_predictions = model.predict(X_test, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    test_accuracy = np.mean(test_pred_classes == y_test_enc)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    class_names = label_encoder.classes_
    
    print(f"\nCLASSIFICATION REPORT:")
    print("-" * 60)
    report = classification_report(y_test_enc, test_pred_classes, 
                                 target_names=class_names, 
                                 digits=4)
    print(report)
    
    cm = confusion_matrix(y_test_enc, test_pred_classes)

    return model, label_encoder, history, X_test, y_test_enc, test_pred_classes, cm


def train_cnn_model_custom_only():
    """Trenuj model TYLKO na własnych danych"""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    
    print("=== TRENING TYLKO NA WŁASNYCH DANYCH ===")
    custom_images, custom_labels = load_custom_gesture_data()
    
    if len(custom_images) == 0:
        print("Brak własnych danych do trenowania!")
        return None
    
    print(f"Wczytano {len(custom_images)} własnych obrazów")

    X_train, X_temp, y_train, y_temp = train_test_split(
        custom_images, custom_labels, test_size=0.3, random_state=42, stratify=custom_labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Podział danych:")
    print(f"   - Train: {len(X_train)} obrazów ({len(X_train)/len(custom_images)*100:.1f}%)")
    print(f"   - Val: {len(X_val)} obrazów ({len(X_val)/len(custom_images)*100:.1f}%)")
    print(f"   - Test: {len(X_test)} obrazów ({len(X_test)/len(custom_images)*100:.1f}%)")
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"\nRozkład klas w treningu:")
    for label, count in zip(unique_train, counts_train):
        print(f"   - {label}: {count} obrazów")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), 
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        PrintEpochMetrics()
    ]

    model = create_cnn_model(X_train.shape[1:], len(set(y_train)))

    history = model.fit(
        datagen.flow(X_train, y_train_enc, batch_size=16), 
        validation_data=(X_val, y_val_enc),
        epochs=30,  
        callbacks=callbacks,
        verbose=1
    )
    print("\n" + "="*60)
    print("FINALNA EWALUACJA MODELU (TYLKO WŁASNE DANE)")
    print("="*60)

    test_predictions = model.predict(X_test, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    test_accuracy = np.mean(test_pred_classes == y_test_enc)
    print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    class_names = label_encoder.classes_
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print("-" * 60)
    report = classification_report(y_test_enc, test_pred_classes, 
                                 target_names=class_names, 
                                 digits=4)
    print(report)
    
    cm = confusion_matrix(y_test_enc, test_pred_classes)

    return model, label_encoder, history, X_test, y_test_enc, test_pred_classes, cm


if __name__ == "__main__":
    import seaborn as sns 
    
    result = train_cnn_model_custom_only()
    
    if result is None:
        print("Błąd trenowania!")
        exit()
    
    model, label_encoder, history, X_test, y_test_enc, test_pred_classes, cm = result

    os.makedirs('../models/CNN', exist_ok=True)
    model.save('../models/CNN/cnn_custom3_only_model.keras')
    with open('../models/CNN/label_encoder_custom3_only.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\nModel (tylko własne dane) został zapisany!")

    print("\nGenerowanie wykresów...")
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Trening', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Walidacja', linewidth=2)
    plt.title('Dokładność modelu\n(Tylko własne dane)', fontsize=14)
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Trening', linewidth=2)
    plt.plot(history.history['val_loss'], label='Walidacja', linewidth=2)
    plt.title('Funkcja straty\n(Tylko własne dane)', fontsize=14)
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    
    test_accuracy = np.mean(test_pred_classes == y_test_enc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Liczba próbek'})
    
    plt.title(f'Macierz Pomyłek (Custom Only)\nTest Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)', 
              fontsize=14)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywistość')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('cnn_custom3_only_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model (tylko własne dane) gotowy!")