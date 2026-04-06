"""
Voice Model - CNN-LSTM for Audio Feature Processing
"""

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


class VoiceModel:
    """Model for processing voice/audio data using CNN-LSTM"""

    def __init__(self, n_mfcc=40, max_length=500):
        """
        Initialize voice model

        Args:
            n_mfcc: Number of MFCC coefficients
            max_length: Maximum sequence length
        """
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        self.model = None

    def extract_mfcc_features(self, audio_path):
        """
        Extract MFCC features from audio file

        Args:
            audio_path: Path to audio file

        Returns:
            MFCC features as numpy array
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=16000)

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )

            # Transpose to get (time_steps, n_mfcc)
            mfccs = mfccs.T

            # Pad or truncate to max_length
            if mfccs.shape[0] < self.max_length:
                # Pad with zeros
                pad_width = self.max_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            else:
                # Truncate
                mfccs = mfccs[:self.max_length, :]

            # Normalize
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)

            return mfccs

        except Exception as e:
            logger.error(f"MFCC extraction error: {str(e)}")
            raise

    def build_model(self):
        """Build CNN-LSTM architecture"""

        model = keras.Sequential([
            # Input layer
            keras.Input(shape=(self.max_length, self.n_mfcc, 1)),

            # CNN layers for spatial feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Reshape for LSTM
            layers.Reshape((-1, 128)),

            # LSTM layers for temporal feature extraction
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),

            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.3),

            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            # Embedding layer (for fusion)
            layers.Dense(32, activation='relu', name='embedding'),

            # Output layer
            layers.Dense(2, activation='softmax', name='output')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the voice model

        Args:
            X_train: Training MFCC features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            self.build_model()

        # Add channel dimension
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Train
        logger.info("Training voice model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return history

    def predict(self, audio_path):
        """
        Get embedding from voice data

        Args:
            audio_path: Path to audio file

        Returns:
            Embedding vector
        """
        # Extract features
        mfccs = self.extract_mfcc_features(audio_path)
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
        mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension

        # Get embedding from intermediate layer
        embedding_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('embedding').output
        )

        embedding = embedding_model.predict(mfccs, verbose=0)[0]
        return embedding

    def get_probability(self, audio_path):
        """
        Get ASD probability prediction

        Args:
            audio_path: Path to audio file

        Returns:
            Probability of ASD (class 1)
        """
        mfccs = self.extract_mfcc_features(audio_path)
        mfccs = np.expand_dims(mfccs, axis=0)
        mfccs = np.expand_dims(mfccs, axis=-1)

        prediction = self.model.predict(mfccs, verbose=0)[0]
        return prediction[1]  # Probability of ASD class

    def save(self, path):
        """Save model"""
        self.model.save(path)
        logger.info(f"Voice model saved to {path}")

    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Voice model loaded from {path}")
