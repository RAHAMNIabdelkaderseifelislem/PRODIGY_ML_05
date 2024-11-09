"""
Data Preprocessor utility module
================================

This module provides a utility class, `DataPreprocessor`, to preprocess a
dataset of images into training and validation sets. The class provides methods
to create necessary directories, split and process images, and create data
generators for training and validation. The class also provides a method to
create a data generator for testing.

The class uses the `ImageDataGenerator` class from `tensorflow.keras.preprocessing`
to generate batches of images for training and validation.

"""
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from .utils import CLASSES

class DataPreprocessor:
    """
    A utility class to preprocess a dataset of images.
    """
    def __init__(self, data_dir, output_dir):
        """
        Initialize the DataPreprocessor object.

        Parameters:
        - data_dir: The path to the dataset of images.
        - output_dir: The path to the output directory where the preprocessed
            data will be saved.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'validation')
        self.target_size = (224, 224)

    def setup_directories(self):
        """Create necessary directories"""
        for class_name in CLASSES:
            os.makedirs(os.path.join(self.train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, class_name), exist_ok=True)

    def process_images(self):
        """
        Process and split images into train/validation sets.

        This method iterates over each class specified in CLASSES, processes the images,
        and splits them into training and validation sets. It warns if the class directory
        does not exist or contains no images. The images are then processed and copied
        to their respective directories.
        """
        for class_name in CLASSES:
            print(f"Processing {class_name}...")
            class_path = os.path.join(self.data_dir, class_name)

            # Check if class directory exists
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist!")
                continue

            # List all image files in the directory
            images = [f for f in os.listdir(class_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]

            # Warn if no images are found in the directory
            if not images:
                print(f"Warning: No images found in {class_path}")
                continue

            # Split images into training and validation sets
            train_images, val_images = train_test_split(
                images, test_size=0.2, random_state=42
            )

            # Process and copy images to respective directories
            for subset, image_list in [('train', train_images), ('validation', val_images)]:
                for img_name in tqdm(image_list, desc=f"{subset.capitalize()} images for {class_name}"):
                    self._process_and_copy_image(
                        class_path, img_name, class_name, subset
                    )

    def _process_and_copy_image(self, src_dir, img_name, class_name, subset):
        """
        Process an individual image by converting it to RGB, resizing, and copying it
        to the appropriate directory based on the subset (train/validation).

        Parameters:
        - src_dir: Source directory of the images.
        - img_name: Name of the image file.
        - class_name: Class label of the image.
        - subset: Subset label ('train' or 'validation') to determine destination.
        """
        try:
            # Construct the path to the source image
            img_path = os.path.join(src_dir, img_name)
            
            # Open the image
            img = Image.open(img_path)
            
            # Convert image to RGB
            img = img.convert('RGB')
            
            # Resize image to target size
            img = img.resize(self.target_size)
            
            # Determine destination directory based on subset
            dest_dir = self.train_dir if subset == 'train' else self.val_dir
            
            # Construct the path to the destination file
            dest_path = os.path.join(dest_dir, class_name, img_name)
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Save the processed image to the destination path
            img.save(dest_path)
            
        except Exception as e:
            # Log any errors that occur during processing
            print(f"Error processing {img_name}: {str(e)}")

    def create_data_generators(self, batch_size=32):
        """
        Create data generators for training and validation.

        The data generators are created using the ImageDataGenerator class from
        tensorflow.keras.preprocessing.image. The training data generator is
        configured to perform random rotation, width shift, height shift, and
        horizontal flip. The validation data generator is configured to simply
        rescale the images.

        Parameters:
        - batch_size: The batch size of the generated data.

        Returns:
        - train_generator: The data generator for training.
        - val_generator: The data generator for validation.
        """
        train_datagen = ImageDataGenerator(
            # Rescale images to the range [0, 1]
            rescale=1./255,
            # Rotate images randomly up to 20 degrees
            rotation_range=20,
            # Shift images randomly up to 20% horizontally and vertically
            width_shift_range=0.2,
            height_shift_range=0.2,
            # Flip images horizontally
            horizontal_flip=True,
            # Fill mode for interpolation
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(
            # Rescale images to the range [0, 1]
            rescale=1./255
        )

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            # Classes is a list of class labels
            classes=CLASSES
        )

        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            # Classes is a list of class labels
            classes=CLASSES
        )

        return train_generator, val_generator

