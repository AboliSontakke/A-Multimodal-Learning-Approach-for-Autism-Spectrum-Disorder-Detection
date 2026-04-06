import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


class FusionModel:

    def __init__(self, behavioral_dim=2, voice_dim=32, facial_dim=64):
       
        self.behavioral_dim = behavioral_dim
        self.voice_dim = voice_dim
        self.facial_dim = facial_dim
        self.total_dim = behavioral_dim + voice_dim + facial_dim
        self.model = None

    def build_model(self):

        combined_input = keras.Input(shape=(self.total_dim,), name='combined_input')

        x = layers.Dense(128, activation='relu')(combined_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(2, activation='softmax', name='output')(x)

        model = keras.Model(inputs=combined_input, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):

        if self.model is None:
            self.build_model()

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )

        logger.info("Training fusion model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return history

    def predict(self, combined_embedding):

        embedding = np.expand_dims(combined_embedding, axis=0)
        prediction = self.model.predict(embedding, verbose=0)[0]
        return np.argmax(prediction)

    def get_probability(self, combined_embedding):

        embedding = np.expand_dims(combined_embedding, axis=0)
        prediction = self.model.predict(embedding, verbose=0)[0]
        return prediction[1]  # Probability of ASD class

    def save(self, path):
        """Save model"""
        self.model.save(path)
        logger.info(f"Fusion model saved to {path}")

    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Fusion model loaded from {path}")


# Alternative: Attention-based Fusion Model
class AttentionFusionModel(FusionModel):

    def build_model(self):

        # Separate inputs for each modality
        behavioral_input = keras.Input(shape=(self.behavioral_dim,), name='behavioral')
        voice_input = keras.Input(shape=(self.voice_dim,), name='voice')
        facial_input = keras.Input(shape=(self.facial_dim,), name='facial')

        # Transform each modality to same dimension
        common_dim = 64

        behavioral_transformed = layers.Dense(common_dim, activation='relu')(behavioral_input)
        voice_transformed = layers.Dense(common_dim, activation='relu')(voice_input)
        facial_transformed = layers.Dense(common_dim, activation='relu')(facial_input)

        # Stack modalities
        stacked = layers.Lambda(lambda x: tf.stack(x, axis=1))(
            [behavioral_transformed, voice_transformed, facial_transformed]
        )

        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=common_dim
        )(stacked, stacked)

        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(attention_output)

        # Dense layers
        x = layers.Dense(128, activation='relu')(pooled)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output
        outputs = layers.Dense(2, activation='softmax')(x)

        # Create model
        model = keras.Model(
            inputs=[behavioral_input, voice_input, facial_input],
            outputs=outputs
        )

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )

        self.model = model
        return model
