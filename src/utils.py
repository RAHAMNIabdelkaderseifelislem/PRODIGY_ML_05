"""
This utility module provides functions for image preprocessing, calorie estimation,
and prediction result formatting for a food classification model. The predefined
calorie dictionary and class labels are specific to this use case.
"""

import numpy as np

# Predefined calorie dictionary for our 5 classes
CALORIE_DICT = {
    'apple_pie': 237,      # per 100g
    'caesar_salad': 127,   # per 100g
    'pizza': 266,          # per 100g
    'sushi': 150,          # per 100g
    'ice_cream': 207,      # per 100g
}

# Class labels
CLASSES = ['apple_pie', 'caesar_salad', 'pizza', 'sushi', 'ice_cream']

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the input image for model input by resizing, normalizing,
    and adding a batch dimension.

    Parameters:
    - image: Input image to be preprocessed.
    - target_size: Tuple specifying the desired image size (width, height).

    Returns:
    - image_array: A preprocessed image array ready for model input.
    """
    # Resize image
    image = image.resize(target_size)
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def get_calorie_estimation(food_class):
    """
    Get calorie estimation for the predicted food class.

    Parameters:
    - food_class: The class label of the predicted food item.

    Returns:
    - Calorie estimation per 100g for the given food class.
    """
    return CALORIE_DICT.get(food_class, 'Unknown')

def format_prediction(prediction, class_name, calories):
    """
    Format the prediction results into a structured dictionary.

    Parameters:
    - prediction: The confidence score of the prediction.
    - class_name: The class label of the predicted food item.
    - calories: The estimated calories for the predicted class.

    Returns:
    - A dictionary containing the formatted prediction results.
    """
    return {
        'class': class_name,
        'confidence': f'{prediction * 100:.2f}%',
        'calories': f'{calories} kcal per 100g'
    }
