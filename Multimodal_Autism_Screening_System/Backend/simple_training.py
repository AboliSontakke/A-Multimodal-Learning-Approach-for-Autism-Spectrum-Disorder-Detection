import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")

# Create necessary directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("\n" + "="*60)
print("MULTIMODAL ASD SCREENING - TRAINING PIPELINE")
print("="*60)

# ==================== BEHAVIORAL MODEL ====================
print("\n[1/3] BEHAVIORAL MODEL (XGBoost)")
print("-" * 60)

BEHAVIORAL_DATA_PATH = 'data/behavioral_data.csv'

if not os.path.exists(BEHAVIORAL_DATA_PATH):
    print("Creating sample behavioral data...")
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = y
    df['id'] = range(n_samples)
    df.to_csv(BEHAVIORAL_DATA_PATH, index=False)
    print(f"✓ Created sample data with {n_samples} samples")

behavioral_df = pd.read_csv(BEHAVIORAL_DATA_PATH)
X_behavioral = behavioral_df.drop(['label', 'id'], axis=1, errors='ignore')
y = behavioral_df['label'].values

scaler = StandardScaler()
X_behavioral_scaled = scaler.fit_transform(X_behavioral)

X_beh_train, X_beh_test, y_beh_train, y_beh_test = train_test_split(
    X_behavioral_scaled, y, test_size=0.2, random_state=42, stratify=y
)

from xgboost import XGBClassifier
behavioral_model = XGBClassifier(n_estimators=50, max_depth=5, random_state=42, verbosity=0)
behavioral_model.fit(X_beh_train, y_beh_train)
y_pred = behavioral_model.predict(X_beh_test)

acc = accuracy_score(y_beh_test, y_pred)
print(f"Accuracy: {acc:.4f}")

with open('saved_models/behavioral_model.pkl', 'wb') as f:
    pickle.dump({'model': behavioral_model, 'scaler': scaler}, f)
print("✓ Model saved to saved_models/behavioral_model.pkl")

# ==================== VOICE MODEL ====================
print("\n[2/3] VOICE MODEL (CNN)")
print("-" * 60)

print("Creating dummy voice data...")
n_samples = 100
X_voice = np.random.randn(n_samples, 500, 40, 1).astype('float32')
y_voice = np.random.randint(0, 2, n_samples)

X_voice_train, X_voice_test, y_voice_train, y_voice_test = train_test_split(
    X_voice, y_voice, test_size=0.2, random_state=42, stratify=y_voice
)

y_voice_train_cat = keras.utils.to_categorical(y_voice_train, 2)
y_voice_test_cat = keras.utils.to_categorical(y_voice_test, 2)

voice_model = keras.Sequential([
    keras.Input(shape=(500, 40, 1)),
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu', name='embedding'),
    keras.layers.Dense(2, activation='softmax')
])

voice_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
voice_model.fit(
    X_voice_train, y_voice_train_cat,
    validation_split=0.2,
    epochs=5,
    batch_size=16,
    verbose=0
)

y_pred = np.argmax(voice_model.predict(X_voice_test, verbose=0), axis=1)
acc = accuracy_score(y_voice_test, y_pred)
print(f"Accuracy: {acc:.4f}")

voice_model.save('saved_models/voice_model.h5')
print("✓ Model saved to saved_models/voice_model.h5")

# ==================== FACIAL MODEL ====================
print("\n[3/3] FACIAL MODEL (CNN)")
print("-" * 60)

print("Creating dummy facial data...")
n_samples = 100
X_img = np.random.randn(n_samples, 128, 128, 3).astype('float32')
y_img = np.random.randint(0, 2, n_samples)

X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42, stratify=y_img
)

y_img_train_cat = keras.utils.to_categorical(y_img_train, 2)
y_img_test_cat = keras.utils.to_categorical(y_img_test, 2)

facial_model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),
    keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu', name='embedding'),
    keras.layers.Dense(2, activation='softmax')
])

facial_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
facial_model.fit(
    X_img_train, y_img_train_cat,
    validation_split=0.2,
    epochs=5,
    batch_size=16,
    verbose=0
)

y_pred = np.argmax(facial_model.predict(X_img_test, verbose=0), axis=1)
acc = accuracy_score(y_img_test, y_pred)
print(f"Accuracy: {acc:.4f}")

facial_model.save('saved_models/facial_model.h5')
print("✓ Model saved to saved_models/facial_model.h5")

# ==================== FUSION MODEL (SIMPLE) ====================
print("\n[BONUS] FUSION MODEL (Dense Network)")
print("-" * 60)

print("Creating fusion embeddings...")
# Get embeddings from each model
beh_emb = behavioral_model.predict_proba(X_beh_test)  # Shape: (n, 2)
voice_emb = np.random.randn(len(y_voice_test), 16)     # Dummy embeddings
facial_emb = np.random.randn(len(y_img_test), 16)      # Dummy embeddings

# Ensure same sample size
min_size = min(len(beh_emb), len(voice_emb), len(facial_emb))
X_fusion = np.concatenate([beh_emb[:min_size], voice_emb[:min_size], facial_emb[:min_size]], axis=1)
y_fusion = y_beh_test[:min_size]
y_fusion_cat = keras.utils.to_categorical(y_fusion, 2)

print(f"Fusion input shape: {X_fusion.shape}")

fusion_model = keras.Sequential([
    keras.Input(shape=(X_fusion.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
fusion_model.fit(
    X_fusion, y_fusion_cat,
    validation_split=0.2,
    epochs=5,
    batch_size=16,
    verbose=0
)

fusion_model.save('saved_models/fusion_model.h5')
print("✓ Model saved to saved_models/fusion_model.h5")

# ==================== COMPLETION ====================
print("\n" + "="*60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nModels saved:")
print("  ✓ saved_models/behavioral_model.pkl")
print("  ✓ saved_models/voice_model.h5")
print("  ✓ saved_models/facial_model.h5")
print("  ✓ saved_models/fusion_model.h5")
print("\nYour Flask API will now load these models!")
print("="*60 + "\n")
