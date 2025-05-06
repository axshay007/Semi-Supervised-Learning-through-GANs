import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, unlabeled_dir, target_size=(56, 56), 
                         mb_size_labeled=32, mb_size_unlabeled=128):
    """
    Create data generators for labeled and unlabeled data from HAM10000 dataset.
    
    Args:
        train_dir: Directory containing labeled training data (benign/malignant)
        unlabeled_dir: Directory containing unlabeled data
        target_size: Size to resize images to
        mb_size_labeled: Mini-batch size for labeled data
        mb_size_unlabeled: Mini-batch size for unlabeled data
        
    Returns:
        tuple: (labeled_generator, unlabeled_generator)
    """
    # Data augmentation for labeled data
    labeled_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Simple rescaling for unlabeled data
    unlabeled_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generator for labeled data
    labeled_generator = labeled_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=mb_size_labeled,
        class_mode='binary',
        shuffle=True
    )
    
    # Create generator for unlabeled data
    unlabeled_generator = unlabeled_datagen.flow_from_directory(
        unlabeled_dir,
        target_size=target_size,
        batch_size=mb_size_unlabeled,
        class_mode=None,  # No labels for unlabeled data
        shuffle=True
    )
    
    return labeled_generator, unlabeled_generator

def create_test_generator(test_dir, target_size=(56, 56), batch_size=32):
    """
    Create data generator for test data.
    
    Args:
        test_dir: Directory containing test data
        target_size: Size to resize images to
        batch_size: Batch size
        
    Returns:
        ImageDataGenerator: Test data generator
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator

def normalize_images(images):
    """
    Normalize images to [-1, 1] range for GAN training.
    
    Args:
        images: Input images in [0, 1] range
        
    Returns:
        Images normalized to [-1, 1] range
    """
    return (images * 2) - 1

def denormalize_images(images):
    """
    Denormalize images from [-1, 1] to [0, 1] range.
    
    Args:
        images: Input images in [-1, 1] range
        
    Returns:
        Images normalized to [0, 1] range
    """
    return (images + 1) / 2 