"""
Behavioral Model - XGBoost/Random Forest for Questionnaire Data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BehavioralModel:
    """Model for processing behavioral questionnaire data"""

    def __init__(self, model_type='xgboost'):
        """
        Initialize behavioral model

        Args:
            model_type: 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()

        if model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

    def preprocess(self, data_path):
        """
        Preprocess behavioral data

        Args:
            data_path: Path to CSV or JSON file with questionnaire responses

        Returns:
            Preprocessed features as numpy array
        """
        try:
            # Read data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format")

            # Remove label column if present
            if 'label' in df.columns:
                df = df.drop('label', axis=1)
            if 'ASD' in df.columns:
                df = df.drop('ASD', axis=1)

            # Handle missing values
            df = df.fillna(df.mean())

            # Convert to numpy array
            features = df.values

            # Scale features
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            else:
                features = self.scaler.fit_transform(features)

            return features

        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train(self, X_train, y_train):
        """
        Train the behavioral model

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete")

    def predict(self, data_path):
        """
        Get embedding/features from behavioral data

        Args:
            data_path: Path to data file

        Returns:
            Feature vector (embedding)
        """
        features = self.preprocess(data_path)

        # For tree-based models, use leaf indices as embeddings
        if self.model_type == 'xgboost':
            embedding = self.model.predict_proba(features)[0]
        else:
            # For Random Forest, use probability predictions
            embedding = self.model.predict_proba(features)[0]

        return embedding

    def get_probability(self, data_path):
        """
        Get ASD probability prediction

        Args:
            data_path: Path to data file

        Returns:
            Probability of ASD (class 1)
        """
        features = self.preprocess(data_path)
        prob = self.model.predict_proba(features)[0][1]
        return prob

    def save(self, path):
        """Save model and scaler"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """Load model and scaler"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
        logger.info(f"Model loaded from {path}")
