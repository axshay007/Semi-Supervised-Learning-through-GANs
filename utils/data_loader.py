import tensorflow as tf
import os
import cv2
import numpy as np
from glob import glob

def load_and_preprocess_image(image_path, target_size=(56, 56)):
    """Load and preprocess a single image."""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) - 127.5) / 127.5
    
    return img

def create_labeled_dataset(data_dir, batch_size=32, target_size=(56, 56)):
    """Create dataset from labeled images."""
    # Get all image paths and labels
    benign_paths = glob(os.path.join(data_dir, 'labeled', 'benign', '*.jpg'))
    malignant_paths = glob(os.path.join(data_dir, 'labeled', 'malignant', '*.jpg'))
    
    image_paths = benign_paths + malignant_paths
    labels = [0] * len(benign_paths) + [1] * len(malignant_paths)
    
    # Convert to numpy arrays
    images = np.array([load_and_preprocess_image(path, target_size) for path in image_paths])
    labels = np.array(labels)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_unlabeled_dataset(data_dir, batch_size=32, target_size=(56, 56)):
    """Create dataset from unlabeled images."""
    # Get all image paths
    image_paths = glob(os.path.join(data_dir, 'unlabeled', '*.jpg'))
    
    # Convert to numpy arrays
    images = np.array([load_and_preprocess_image(path, target_size) for path in image_paths])
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def apply_augmentation(image):
    """Apply data augmentation to images."""
    # Random flip left-right
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Random rotation
    angle = tf.random.uniform((), minval=-0.25, maxval=0.25)
    image = tf.image.rotate(image, angle)
    
    # Random brightness
    image = tf.image.random_brightness(image, 0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Ensure values are in [-1, 1]
    image = tf.clip_by_value(image, -1.0, 1.0)
    
    return image

def create_augmented_dataset(dataset):
    """Apply augmentation to a dataset."""
    return dataset.map(
        lambda x, y: (apply_augmentation(x), y) if isinstance(x, tuple) else apply_augmentation(x),
        num_parallel_calls=tf.data.AUTOTUNE
    ) 