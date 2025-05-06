import tensorflow as tf
from tensorflow.keras import layers, Model

class Discriminator(Model):
    """
    Discriminator Network for Semi-Supervised GAN
    
    This class implements a multi-task discriminator that performs both:
    1. Real/Fake image classification (GAN task)
    2. Melanoma detection (supervised task)
    
    The network uses shared convolutional layers followed by two separate
    classification heads for each task. This architecture enables the model
    to learn both tasks simultaneously while sharing feature extraction.
    
    Architecture:
    - Input: 56x56x3 RGB image
    - Shared layers: Series of convolutional blocks with increasing channels
    - Two output heads:
        * GAN head: Real/Fake classification
        * Melanoma head: Benign/Malignant classification
    """
    
    def __init__(self):
        """Initialize the Discriminator network with shared layers and dual heads."""
        super(Discriminator, self).__init__()
        
        # Shared convolutional layers for feature extraction
        self.shared_layers = tf.keras.Sequential([
            # First conv block: 56x56x3 -> 28x28x32
            layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),  # Prevent overfitting
            
            # Second conv block: 28x28x32 -> 14x14x64
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # Third conv block: 14x14x64 -> 7x7x128
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # Flatten features for dense layers
            layers.Flatten()
        ])
        
        # GAN classification head (Real/Fake)
        self.gan_classifier = tf.keras.Sequential([
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Melanoma classification head (Benign/Malignant)
        self.melanoma_classifier = tf.keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.4),  # Additional dropout for robustness
            layers.Dense(2, activation='softmax')  # Binary classification with probabilities
        ])
    
    def call(self, inputs, training=True):
        """
        Forward pass of the discriminator.
        
        Args:
            inputs: Input images of shape (batch_size, 56, 56, 3)
            training (bool): Whether the model is in training mode
            
        Returns:
            tuple: (gan_output, melanoma_output)
                - gan_output: Real/Fake predictions
                - melanoma_output: Benign/Malignant predictions
        """
        features = self.shared_layers(inputs, training=training)
        gan_output = self.gan_classifier(features)
        melanoma_output = self.melanoma_classifier(features)
        return gan_output, melanoma_output
    
    @tf.function
    def train_step_gan(self, real_images, fake_images, optimizer):
        """
        Training step for GAN classification task.
        
        Args:
            real_images: Batch of real images
            fake_images: Batch of generated images
            optimizer: Optimizer for updating weights
            
        Returns:
            float: Total GAN loss for this step
        """
        with tf.GradientTape() as tape:
            # Classify real images (target: 1)
            real_gan_output, _ = self(real_images, training=True)
            real_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_gan_output), real_gan_output)
            
            # Classify fake images (target: 0)
            fake_gan_output, _ = self(fake_images, training=True)
            fake_loss = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_gan_output), fake_gan_output)
            
            total_loss = real_loss + fake_loss
            
        # Update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss
    
    @tf.function
    def train_step_melanoma(self, images, labels, optimizer):
        """
        Training step for melanoma classification task.
        
        Args:
            images: Batch of labeled images
            labels: Ground truth labels (0: benign, 1: malignant)
            optimizer: Optimizer for updating weights
            
        Returns:
            float: Classification loss for this step
        """
        with tf.GradientTape() as tape:
            # Get melanoma predictions
            _, melanoma_output = self(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels, melanoma_output)
            
        # Update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss 