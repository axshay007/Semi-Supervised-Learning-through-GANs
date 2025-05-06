import tensorflow as tf
import os
import cv2
import numpy as np
from glob import glob

def load_and_preprocess_image(image_path, target_size=(56, 56)):
    """
    Load and preprocess a single image for the GAN model.
    
    This function:
    1. Loads an image from disk
    2. Converts it to RGB format
    3. Resizes it to the target size
    4. Normalizes pixel values to [-1, 1] range
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Read image in BGR format
    img = cv2.imread(image_path)
    # Convert to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [-1, 1] range for GAN training
    img = (img.astype(np.float32) - 127.5) / 127.5
    
    return img

def create_labeled_dataset(data_dir, batch_size=32, target_size=(56, 56)):
    """
    Create a TensorFlow dataset from labeled melanoma images.
    
    This function:
    1. Loads images from benign and malignant directories
    2. Creates corresponding labels (0 for benign, 1 for malignant)
    3. Combines them into a shuffled and batched dataset
    
    Args:
        data_dir (str): Root directory containing labeled data
        batch_size (int): Number of images per batch
        target_size (tuple): Target size for images
        
    Returns:
        tf.data.Dataset: Dataset of (image, label) pairs
    """
    # Get paths for benign and malignant images
    benign_paths = glob(os.path.join(data_dir, 'labeled', 'benign', '*.jpg'))
    malignant_paths = glob(os.path.join(data_dir, 'labeled', 'malignant', '*.jpg'))
    
    # Combine paths and create labels
    image_paths = benign_paths + malignant_paths
    labels = [0] * len(benign_paths) + [1] * len(malignant_paths)
    
    # Load and preprocess all images
    images = np.array([load_and_preprocess_image(path, target_size) 
                      for path in image_paths])
    labels = np.array(labels)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images))  # Randomize order
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize performance
    
    return dataset

def create_unlabeled_dataset(data_dir, batch_size=32, target_size=(56, 56)):
    """
    Create a TensorFlow dataset from unlabeled images.
    
    Args:
        data_dir (str): Root directory containing unlabeled data
        batch_size (int): Number of images per batch
        target_size (tuple): Target size for images
        
    Returns:
        tf.data.Dataset: Dataset of unlabeled images
    """
    # Get all unlabeled image paths
    image_paths = glob(os.path.join(data_dir, 'unlabeled', '*.jpg'))
    
    # Load and preprocess images
    images = np.array([load_and_preprocess_image(path, target_size) 
                      for path in image_paths])
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def apply_augmentation(image):
    """
    Apply data augmentation to an image.
    
    This function applies the following augmentations:
    1. Random horizontal flip
    2. Random rotation
    3. Random brightness adjustment
    4. Random contrast adjustment
    
    Args:
        image: Input image tensor
        
    Returns:
        tf.Tensor: Augmented image
    """
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Random rotation (±90 degrees)
    angle = tf.random.uniform((), minval=-0.25, maxval=0.25)
    image = tf.image.rotate(image, angle)
    
    # Random brightness adjustment
    image = tf.image.random_brightness(image, 0.2)
    
    # Random contrast adjustment
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Ensure values remain in [-1, 1] range
    image = tf.clip_by_value(image, -1.0, 1.0)
    
    return image

def create_augmented_dataset(dataset):
    """
    Apply data augmentation to a dataset.
    
    Args:
        dataset: Input TensorFlow dataset
        
    Returns:
        tf.data.Dataset: Augmented dataset
    """
    return dataset.map(
        lambda x, y: (apply_augmentation(x), y) if isinstance(x, tuple) else apply_augmentation(x),
        num_parallel_calls=tf.data.AUTOTUNE
    ) 