
"""
Facial Model - ResNet50/EfficientNet for Facial Image Processing
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


class FacialModel:
    """Model for processing facial images using pretrained CNNs"""

    def __init__(self, model_type='resnet50', img_size=(224, 224)):
        """
        Initialize facial model

        Args:
            model_type: 'resnet50' or 'efficientnet'
            img_size: Input image size
        """
        self.model_type = model_type
        self.img_size = img_size
        self.model = None

    def preprocess_image(self, image_path):
        """
        Preprocess facial image

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array
        """
        try:
            # Read image
            img = Image.open(image_path).convert('RGB')

            # Resize
            img = img.resize(self.img_size)

            # Convert to array
            img_array = np.array(img)

            # Normalize based on model type
            if self.model_type == 'resnet50':
                img_array = keras.applications.resnet50.preprocess_input(img_array)
            elif self.model_type == 'efficientnet':
                img_array = keras.applications.efficientnet.preprocess_input(img_array)
            else:
                img_array = img_array / 255.0

            return img_array

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise

    def build_model(self):
    """High-accuracy EfficientNet model (optimized for small dataset)"""

    base_model = tf.keras.applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze most layers (important for small dataset)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))

    # Strong augmentation (helps accuracy a LOT)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.1)(x)

    # Base model
    x = base_model(x, training=False)

    # Better pooling
    x = layers.GlobalAveragePooling2D()(x)

    # 🔥 Strong classifier head
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # Embedding (important for fusion)
    embedding = layers.Dense(128, activation='relu', name='embedding')(x)

    outputs = layers.Dense(2, activation='softmax')(embedding)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    self.model = model
    self.base_model = base_model
    return model

    def train(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=16):

    if self.model is None:
        self.build_model()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-7
    )

    print("🔥 Phase 1: Train top layers")
    self.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("🔥 Phase 2: Fine-tuning full model")

    # Unfreeze last layers only
    for layer in self.base_model.layers[-30:]:
        layer.trainable = True

    self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    self.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    def predict(self, image_path):
        """
        Get embedding from facial image

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector
        """
        # Preprocess image
        img = self.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Get embedding from intermediate layer
        embedding_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('embedding').output
        )

        embedding = embedding_model.predict(img, verbose=0)[0]
        return embedding

    def get_probability(self, image_path):
        """
        Get ASD probability prediction

        Args:
            image_path: Path to image file

        Returns:
            Probability of ASD (class 1)
        """
        img = self.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img, verbose=0)[0]
        return prediction[1]  # Probability of ASD class

    def save(self, path):
        """Save model"""
        self.model.save(path)
        logger.info(f"Facial model saved to {path}")

    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        logger.info(f"Facial model loaded from {path}")
