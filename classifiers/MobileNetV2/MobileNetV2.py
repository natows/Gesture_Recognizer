import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_custom_gesture_data_mobilenet(data_path='../../dataset'):
    """Wczytuje dane w rozmiarze 224x224 dla MobileNetV2"""
    images = []
    labels = []
    
    print("Wczytywanie danych dla MobileNetV2 (224x224)...")
    
    if not os.path.exists(data_path):
        print(f"Folder {data_path} nie istnieje")
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
                        image = cv2.resize(image, (224, 224))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
                        
                        images.append(image)
                        labels.append(gesture_folder)
                        image_count += 1
                        
                except Exception as e:
                    print(f"Błąd wczytywania {image_path}: {e}")
        
        print(f"{gesture_folder}: {image_count} obrazów")
    
    return np.array(images), np.array(labels)


def create_mobilenet_model(num_classes):
    """Tworzy model z transfer learning MobileNetV2"""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,           
        weights='imagenet'          
    )
    base_model.trainable = False   

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),    
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def fine_tune_model(model, base_model, num_classes):
    """Fine-tuning - odmrażamy niektóre warstwy"""
    
    base_model.trainable = True

    fine_tune_at = len(base_model.layers) - 50
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_mobilenet_transfer_learning():
    """Trenowanie z transfer learning w dwóch fazach"""

    print("=== TRANSFER LEARNING Z MOBILENETV2 ===")
    images, labels = load_custom_gesture_data_mobilenet()
    
    if len(images) == 0:
        print("Brak danych!")
        return None
    
    print(f"Wczytano {len(images)} obrazów")

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Podział: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    print(f"Klasy: {label_encoder.classes_}")

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,         
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest' 
    )
    
    val_datagen = ImageDataGenerator() 
    
    print("\nFAZA 1: Feature extraction (zamrożone warstwy)")
    model, base_model = create_mobilenet_model(len(label_encoder.classes_))
    model.summary()
    
    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]

    history1 = model.fit(
        train_datagen.flow(X_train, y_train_enc, batch_size=16),
        validation_data=(X_val, y_val_enc),
        epochs=20,
        callbacks=callbacks_phase1,
        verbose=1
    )

    print("\nFAZA 2: Fine-tuning (odmrożone ostatnie warstwy)")
    model = fine_tune_model(model, base_model, len(label_encoder.classes_))
    
    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-8)
    ]
    
    history2 = model.fit(
        train_datagen.flow(X_train, y_train_enc, batch_size=8), 
        validation_data=(X_val, y_val_enc),
        epochs=30,
        callbacks=callbacks_phase2,
        verbose=1
    )

    print("\n" + "="*60)
    print("FINALNA EWALUACJA MOBILENETV2")
    print("="*60)
    
    test_predictions = model.predict(X_test, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_pred_classes == y_test_enc)
    
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print(f"\nCLASSIFICATION REPORT:")
    print("-" * 60)
    report = classification_report(y_test_enc, test_pred_classes, 
                                 target_names=label_encoder.classes_, 
                                 digits=4)
    print(report)

    cm = confusion_matrix(y_test_enc, test_pred_classes)

    total_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    return model, label_encoder, total_history, X_test, y_test_enc, test_pred_classes, cm


def visualize_results(history, label_encoder, test_pred_classes, y_test_enc, cm):
    """Wizualizacja wyników"""
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    epochs_phase1 = len([h for h in history['accuracy'] if h < 0.9])  
    
    plt.plot(history['accuracy'], label='Train', linewidth=2)
    plt.plot(history['val_accuracy'], label='Validation', linewidth=2)
    plt.axvline(x=epochs_phase1, color='red', linestyle='--', alpha=0.7, label='Fine-tuning start')
    plt.title('MobileNetV2 Transfer Learning\nAccuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.axvline(x=epochs_phase1, color='red', linestyle='--', alpha=0.7, label='Fine-tuning start')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    test_accuracy = np.mean(test_pred_classes == y_test_enc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix\nTest Acc: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../classifiers/MobileNetV2/mobilenetv2_extended_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Uruchamianie transfer learning z MobileNetV2...")
    
    result = train_mobilenet_transfer_learning()
    
    if result is None:
        print("Błąd trenowania!")
        exit()
    
    model, label_encoder, history, X_test, y_test_enc, test_pred_classes, cm = result
    
    os.makedirs('../models/MobileNetV2', exist_ok=True)
    model.save('../models/MobileNetV2/mobilenetv2_extended_model.keras')
    with open('../models/MobileNetV2/extended_label_encoder_mobilenet.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Model MobileNetV2 został zapisany!")
    
    visualize_results(history, label_encoder, test_pred_classes, y_test_enc, cm)
    
    print("\nTransfer learning zakończony!")