"""
This module provides a utility class to build, train and save a
FoodClassifier model. The model is based on the EfficientNetB0
architecture and is trained on the food classification dataset.

"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
from .utils import CLASSES

class FoodClassifier:
    """
    A utility class to build, train and save a FoodClassifier model.
    """
    def __init__(self):
        """
        Initialize the FoodClassifier object.
        """
        self.model = None
        self.history = None
        self.num_classes = len(CLASSES)

    def build_model(self, learning_rate=0.001):
        """
        Build and compile the model.
        """
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Freeze the base model
        base_model.trainable = False

        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_generator, val_generator, epochs=20):
        """
        Train the model.
        """
        return self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

    def save_model(self, filepath):
        """
        Save the trained model.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Load a trained model.
        """
        self.model = tf.keras.models.load_model(filepath)
