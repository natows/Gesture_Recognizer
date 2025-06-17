import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


SEQ_LENGTH = 10  
N_LANDMARKS = 21  
N_COORDS = 2     

print(" LSTM Gesture Recognition Training")
print("=" * 50)


print("Wczytywanie danych...")
df = pd.read_csv('../../data_sources/sequences.csv')
print(f"Wczytano {len(df)} sekwencji")


print("\n Rozkład gestów:")
gesture_counts = df['label'].value_counts()
for gesture, count in gesture_counts.items():
    print(f"  {gesture}: {count} sekwencji")


print("\n Przetwarzanie danych...")

feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].values
y = df['label'].values

print(f" Kształt X: {X.shape}")
print(f" Liczba cech na klatkę: {len(feature_columns) // SEQ_LENGTH}")



n_samples = X.shape[0]
n_features_per_frame = len(feature_columns) // SEQ_LENGTH 

X_reshaped = X.reshape(n_samples, SEQ_LENGTH, n_features_per_frame)
print(f" Dane po reshape: {X_reshaped.shape}")


print("\nEnkodowanie etykiet...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f" Klasy: {list(label_encoder.classes_)}")
print(f" Liczba klas: {len(label_encoder.classes_)}")


print("\n Podział danych...")
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42
)

print(f" Trening: {X_train.shape[0]} próbek")
print(f" Walidacja: {X_val.shape[0]} próbek") 
print(f" Test: {X_test.shape[0]} próbek")

print("\n Budowa modelu LSTM...")

model = Sequential([
    LSTM(128, return_sequences=True, 
         input_shape=(SEQ_LENGTH, n_features_per_frame),
         name='lstm_1'),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(64, return_sequences=True, name='lstm_2'),
    BatchNormalization(), 
    Dropout(0.3),
    
    LSTM(32, return_sequences=False, name='lstm_3'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu', name='dense_1'),
    Dropout(0.5),
    
    Dense(32, activation='relu', name='dense_2'),
    Dropout(0.3),
    
    Dense(len(label_encoder.classes_), activation='softmax', name='output')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n Architektura modelu:")
model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n Rozpoczynanie treningu...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print("\n Ewaluacja modelu...")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f" Trening - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
print(f" Walidacja - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
print(f" Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

print("\n Analiza predykcji...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print("\n Raport klasyfikacji:")
print(classification_report(
    y_true_classes, y_pred_classes, 
    target_names=label_encoder.classes_
))

cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\n Macierz pomyłek:")
print(cm)

print("\n Generowanie wykresów...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Walidacja', linewidth=2)
plt.title('Dokładność modelu LSTM', fontsize=14, fontweight='bold')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening', linewidth=2)
plt.plot(history.history['val_loss'], label='Walidacja', linewidth=2)
plt.title('Strata modelu LSTM', fontsize=14, fontweight='bold')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Zapisywanie modelu...")
import os
os.makedirs('models/LSTM', exist_ok=True)

model.save('../../models/LSTM/lstm_gesture_model.keras')
print(" Model zapisany: models/LSTM/lstm_gesture_model.keras")

import pickle
with open('../../models/LSTM/label_encoder_lstm.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(" Label encoder zapisany: models/LSTM/label_encoder_lstm.pkl")

model_info = {
    'sequence_length': SEQ_LENGTH,
    'n_landmarks': N_LANDMARKS,
    'n_coords': N_COORDS,
    'n_features_per_frame': n_features_per_frame,
    'classes': list(label_encoder.classes_),
    'test_accuracy': test_acc,
    'input_shape': X_reshaped.shape[1:]
}

import json
with open('../../models/LSTM/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("Info o modelu zapisane: models/LSTM/model_info.json")

print(f"\nTRENOWANIE ZAKOŃCZONE!")
print(f" Finalna dokładność: {test_acc:.2%}")
print("=" * 50)