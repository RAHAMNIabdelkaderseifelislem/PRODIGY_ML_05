"""
app.py

This file contains the code for the Food Recognition System, a Gradio-based web
application for training and testing a food recognition model. The application
allows users to upload an image of food and get the predicted food class and
estimated calories. The application also provides a training tab for users to
configure and start training the model.

The application uses the EfficientNetB0 model from the TensorFlow Keras
application, and the model is trained on the Food-101 dataset. The application
also uses the PIL library for image processing and the Gradio library for
building the web interface.

The application is designed to be easy to use and does not require any
programming knowledge. Users can simply upload an image of food and get the
predicted food class and estimated calories.

"""

import os
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from src.preprocessing import DataPreprocessor
from src.model import FoodClassifier
from src.utils import CLASSES, preprocess_image, get_calorie_estimation, format_prediction
import numpy as np
import pandas as pd

class FoodRecognitionApp:
    """
    Class for the Food Recognition System
    """

    def __init__(self):
        """
        Initialize the FoodRecognitionApp object
        """
        self.model = None
        self.history = None
        self.data_preprocessor = None
        self.trainer = None

    def train_model(self, data_dir, batch_size, epochs, learning_rate, progress=gr.Progress()):
        """
        Train the model and return training plots
        """
        try:
            # Initialize preprocessor
            self.data_preprocessor = DataPreprocessor(
                data_dir=data_dir,
                output_dir="data/processed"
            )
            
            progress(0.2, desc="Preprocessing data...")
            # Prepare data
            self.data_preprocessor.setup_directories()
            self.data_preprocessor.process_images()
            train_generator, val_generator = self.data_preprocessor.create_data_generators(batch_size)
            
            progress(0.4, desc="Building model...")
            # Initialize and train model
            self.trainer = FoodClassifier()
            self.trainer.build_model(learning_rate)
            
            progress(0.6, desc="Training model...")
            # Train model
            history = self.trainer.train(train_generator, val_generator, epochs)
            self.history = history.history
            
            progress(0.8, desc="Saving model...")
            # Save model
            os.makedirs('models', exist_ok=True)
            self.trainer.save_model('models/food_classifier.h5')
            
            progress(0.9, desc="Generating plots...")
            # Create training plots using plotly
            acc_fig = px.line(
                pd.DataFrame({
                    'Training Accuracy': self.history['accuracy'],
                    'Validation Accuracy': self.history['val_accuracy'],
                    'Epoch': range(1, len(self.history['accuracy']) + 1)
                }),
                x='Epoch',
                y=['Training Accuracy', 'Validation Accuracy'],
                title='Model Accuracy over Time'
            )

            loss_fig = px.line(
                pd.DataFrame({
                    'Training Loss': self.history['loss'],
                    'Validation Loss': self.history['val_loss'],
                    'Epoch': range(1, len(self.history['loss']) + 1)
                }),
                x='Epoch',
                y=['Training Loss', 'Validation Loss'],
                title='Model Loss over Time'
            )
            
            progress(1.0, desc="Done!")
            return (
                acc_fig,
                loss_fig,
                "Training completed successfully! Model saved to 'models/food_classifier.h5'"
            )
            
        except Exception as e:
            return None, None, f"Error during training: {str(e)}"

    def predict(self, image):
        """
        Make prediction on uploaded image
        """
        if self.trainer is None:
            try:
                self.trainer = FoodClassifier()
                self.trainer.load_model('models/food_classifier.h5')
            except Exception as e:
                return "Error: Model not found. Please train the model first."

        try:
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = self.trainer.model.predict(processed_image)[0]
            predicted_class_idx = np.argmax(prediction)
            predicted_class = CLASSES[predicted_class_idx]
            
            # Get calorie estimation
            calories = get_calorie_estimation(predicted_class)
            
            # Format results
            result = format_prediction(
                prediction[predicted_class_idx],
                predicted_class,
                calories
            )
            
            # Create confidence plot
            confidence_fig = px.bar(
                x=CLASSES,
                y=prediction * 100,
                title='Prediction Confidence for Each Class',
                labels={'x': 'Food Class', 'y': 'Confidence (%)'}
            )
            
            return (
                f"Food: {result['class'].replace('_', ' ').title()}\n"
                f"Confidence: {result['confidence']}\n"
                f"Estimated Calories: {result['calories']}",
                confidence_fig
            )
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", None

    def create_interface(self):
        """
        Create Gradio interface with training and testing tabs
        """
        with gr.Blocks(title="🍔 Food Recognition System") as interface:
            gr.Markdown("# 🍕 Food Recognition & Calorie Estimation System")
            
            with gr.Tabs():
                # Training Tab
                with gr.Tab("🎯 Train Model"):
                    gr.Markdown("### Model Training Configuration")
                    with gr.Row():
                        data_dir = gr.Textbox(
                            label="Data Directory",
                            value="data/food101",
                            placeholder="Path to your dataset"
                        )
                        batch_size = gr.Number(
                            label="Batch Size",
                            value=32,
                            minimum=1,
                            maximum=128,
                            step=1
                        )
                    with gr.Row():
                        epochs = gr.Slider(
                            label="Number of Epochs",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1
                        )
                        learning_rate = gr.Slider(
                            label="Learning Rate",
                            minimum=0.0001,
                            maximum=0.01,
                            value=0.001,
                            step=0.0001
                        )
                    train_btn = gr.Button("🚀 Start Training")
                    with gr.Row():
                        acc_plot = gr.Plot(label="Accuracy Plot")
                        loss_plot = gr.Plot(label="Loss Plot")
                    train_output = gr.Textbox(label="Training Status")
                    
                    train_btn.click(
                        fn=self.train_model,
                        inputs=[data_dir, batch_size, epochs, learning_rate],
                        outputs=[acc_plot, loss_plot, train_output]
                    )

                # Testing Tab
                with gr.Tab("🔍 Test Model"):
                    gr.Markdown("### Upload an image to test the model")
                    with gr.Row():
                        image_input = gr.Image(type="pil", label="Upload Food Image")
                        with gr.Column():
                            prediction_output = gr.Textbox(label="Prediction Results")
                            confidence_plot = gr.Plot(label="Confidence Distribution")
                    
                    test_btn = gr.Button("🔍 Analyze Image")
                    test_btn.click(
                        fn=self.predict,
                        inputs=[image_input],
                        outputs=[prediction_output, confidence_plot]
                    )
                    
                    gr.Markdown("### Example Images")
                    gr.Examples(
                        examples=[
                            ["examples/apple_pie.jpg"],
                            ["examples/caesar_salad.jpg"],
                            ["examples/pizza.jpg"],
                        ],
                        inputs=image_input
                    )

        return interface

if __name__ == "__main__":
    app = FoodRecognitionApp()
    interface = app.create_interface()
    interface.launch(share=True)
