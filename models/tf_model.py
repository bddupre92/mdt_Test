"""
tf_model.py
-----------
Example TensorFlow Keras model wrapper for migraine classification.
"""

import numpy as np
import tensorflow as tf

from .base_model import BaseModel

class TFModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=16, lr=0.001, epochs=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self._build_model()
    
    def _build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
    
    def train(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
    
    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        return (preds >= 0.5).astype(int).squeeze()
