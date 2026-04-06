import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import librosa
import cv2
from PIL import Image
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Create necessary directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('data/voice_samples', exist_ok=True)
os.makedirs('data/facial_images', exist_ok=True)

# Load Dataset
BEHAVIORAL_DATA_PATH = 'data/behavioral_data.csv'
VOICE_DATA_PATH = 'data/voice_samples/'
IMAGE_DATA_PATH = 'data/facial_images/'

# Create sample behavioral data if it doesn't exist
if not os.path.exists(BEHAVIORAL_DATA_PATH):
    print(f"Creating sample behavioral data for demonstration...")
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = y
    df['id'] = range(n_samples)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(BEHAVIORAL_DATA_PATH, index=False)
    print(f"✓ Created sample behavioral data at {BEHAVIORAL_DATA_PATH}")

behavioral_df = pd.read_csv(BEHAVIORAL_DATA_PATH)
print(f"Behavioral data shape: {behavioral_df.shape}")

print(f"\nLabel distribution:")
print(behavioral_df['label'].value_counts())

# 2. Data Preprocessing
X_behavioral = behavioral_df.drop(['label', 'id'], axis=1, errors='ignore')
y = behavioral_df['label'].values
X_behavioral = X_behavioral.fillna(X_behavioral.mean())
scaler = StandardScaler()
X_behavioral_scaled = scaler.fit_transform(X_behavioral)
print(f"Behavioral features shape: {X_behavioral_scaled.shape}")

# Create dummy voice data
if not os.path.exists(VOICE_DATA_PATH) or len(os.listdir(VOICE_DATA_PATH)) == 0:
    print(f"Creating dummy voice data...")
    X_voice = np.random.randn(len(y), 500, 40)
    y_voice = y.copy()
else:
    X_voice = np.random.randn(len(y), 500, 40)
    y_voice = y.copy()

print(f"Voice features shape: {X_voice.shape}")

# Create dummy image data
if not os.path.exists(IMAGE_DATA_PATH) or len(os.listdir(IMAGE_DATA_PATH)) == 0:
    print(f"Creating dummy image data...")
    X_images = np.random.randn(len(y), 224, 224, 3)
    y_images = y.copy()
else:
    X_images = np.random.randn(len(y), 224, 224, 3)
    y_images = y.copy()

print(f"Image features shape: {X_images.shape}")

# Ensure all modalities have same number of samples
min_samples = min(len(y), len(y_voice), len(y_images))
print(f"\nAligning all modalities to {min_samples} samples...")

X_behavioral_scaled = X_behavioral_scaled[:min_samples]
y = y[:min_samples]
X_voice = X_voice[:min_samples]
y_voice = y_voice[:min_samples]
X_images = X_images[:min_samples]
y_images = y_images[:min_samples]

print(f"Final shapes - Behavioral: {X_behavioral_scaled.shape}, Voice: {X_voice.shape}, Images: {X_images.shape}")

# 3. Train-Test Split
X_beh_train, X_beh_test, y_beh_train, y_beh_test = train_test_split(
    X_behavioral_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_voice_train, X_voice_test, y_voice_train, y_voice_test = train_test_split(
    X_voice, y_voice, test_size=0.2, random_state=42, stratify=y_voice
)
X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
    X_images, y_images, test_size=0.2, random_state=42, stratify=y_images
)

y_voice_train_cat = keras.utils.to_categorical(y_voice_train, 2)
y_voice_test_cat = keras.utils.to_categorical(y_voice_test, 2)
y_img_train_cat = keras.utils.to_categorical(y_img_train, 2)
y_img_test_cat = keras.utils.to_categorical(y_img_test, 2)

# 4. Behavioral Model (XGBoost)
from xgboost import XGBClassifier
print("\n" + "="*50)
print("Training Behavioral Model...")
print("="*50)
behavioral_model = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
behavioral_model.fit(X_beh_train, y_beh_train)
y_pred = behavioral_model.predict(X_beh_test)
y_pred_proba = behavioral_model.predict_proba(X_beh_test)[:, 1]
print(f"✓ Behavioral Model Accuracy: {accuracy_score(y_beh_test, y_pred):.4f}")
with open('saved_models/behavioral_model.pkl', 'wb') as f:
    pickle.dump({'model': behavioral_model, 'scaler': scaler}, f)
print("✓ Behavioral model saved\n")

# 5. Voice Model (CNN-LSTM)
def build_voice_model(input_shape=(500, 40, 1)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', name='embedding'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X_voice_train_exp = np.expand_dims(X_voice_train, -1)
X_voice_test_exp = np.expand_dims(X_voice_test, -1)
voice_model = build_voice_model()
print("="*50)
print("Training Voice Model...")
print("="*50)
history_voice = voice_model.fit(
    X_voice_train_exp, y_voice_train_cat,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
y_pred = np.argmax(voice_model.predict(X_voice_test_exp, verbose=0), axis=1)
print(f"✓ Voice Model Accuracy: {accuracy_score(y_voice_test, y_pred):.4f}")
voice_model.save('saved_models/voice_model.h5')
print("✓ Voice model saved\n")

# 6. Facial Model (Simple CNN)
def build_facial_model(img_size=(224, 224)):
    model = keras.Sequential([
        keras.Input(shape=(*img_size, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu', name='embedding'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

facial_model = build_facial_model()
print("="*50)
print("Training Facial Model...")
print("="*50)
history_facial = facial_model.fit(
    X_img_train, y_img_train_cat,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
y_pred = np.argmax(facial_model.predict(X_img_test, verbose=0), axis=1)
print(f"✓ Facial Model Accuracy: {accuracy_score(y_img_test, y_pred):.4f}")
facial_model.save('saved_models/facial_model.h5')
print("✓ Facial model saved\n")

# 7. Fusion Model
def get_embeddings(X_beh, X_voice, X_img):
    """Extract embeddings from all models"""
    embeddings = []
    
    # Behavioral embeddings
    beh_emb = behavioral_model.predict_proba(X_beh)
    embeddings.append(beh_emb)
    
    # Voice embeddings - FIXED: Build embedding model correctly
    embedding_model_voice = keras.Model(
        inputs=voice_model.input,
        outputs=voice_model.get_layer('embedding').output
    )
    voice_emb = embedding_model_voice.predict(X_voice, verbose=0)
    embeddings.append(voice_emb)
    
    # Facial embeddings - FIXED: Build embedding model correctly
    embedding_model_facial = keras.Model(
        inputs=facial_model.input,
        outputs=facial_model.get_layer('embedding').output
    )
    facial_emb = embedding_model_facial.predict(X_img, verbose=0)
    embeddings.append(facial_emb)
    
    return np.concatenate(embeddings, axis=1)

print("="*50)
print("Extracting embeddings...")
print("="*50)
# Pass data in correct format (already expanded for voice/img)
X_fusion_train = get_embeddings(X_beh_train, np.expand_dims(X_voice_train, -1), X_img_train)
X_fusion_test = get_embeddings(X_beh_test, np.expand_dims(X_voice_test, -1), X_img_test)

y_fusion_train = keras.utils.to_categorical(y_beh_train, 2)
y_fusion_test = keras.utils.to_categorical(y_beh_test, 2)

print(f"Fusion training data shape: {X_fusion_train.shape}")

def build_fusion_model(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

fusion_model = build_fusion_model(X_fusion_train.shape[1])
print("="*50)
print("Training Fusion Model...")
print("="*50)
history_fusion = fusion_model.fit(
    X_fusion_train, y_fusion_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)
y_pred = np.argmax(fusion_model.predict(X_fusion_test, verbose=0), axis=1)
y_true = np.argmax(y_fusion_test, axis=1)
print("\n" + "="*50)
print("FUSION MODEL RESULTS")
print("="*50)
print(f"✓ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"✓ Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"✓ Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"✓ F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"✓ AUROC: {roc_auc_score(y_true, y_pred):.4f}")
fusion_model.save('saved_models/fusion_model.h5')
print("✓ Fusion model saved\n")

print("="*50)
print("✅ TRAINING COMPLETE!")
print("="*50)
print("\nAll models saved to 'saved_models/' directory:")
print("  ✓ saved_models/behavioral_model.pkl")
print("  ✓ saved_models/voice_model.h5")
print("  ✓ saved_models/facial_model.h5")
print("  ✓ saved_models/fusion_model.h5")
print("\nYour Flask API will now load these models automatically!")
print("="*50)