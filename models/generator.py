import tensorflow as tf
from tensorflow.keras import layers, Model

class Generator(Model):
    """
    Generator Network for Semi-Supervised GAN
    
    This class implements a deep convolutional generator network that creates synthetic
    skin lesion images from random noise. The architecture progressively upsamples
    the input from a latent space to create a 56x56x3 RGB image.
    
    Architecture:
    - Input: Random noise vector of size latent_dim
    - Dense layer to create initial 7x7x256 feature map
    - Progressive upsampling using transposed convolutions:
        * 7x7 -> 14x14 -> 28x28 -> 56x56
    - Final output: 56x56x3 RGB image normalized to [-1, 1]
    """
    
    def __init__(self, latent_dim=100):
        """
        Initialize the Generator network.
        
        Args:
            latent_dim (int): Dimension of the input noise vector (default: 100)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Define the generator architecture
        self.model = tf.keras.Sequential([
            # Initial dense layer to create 7x7x256 feature map from noise
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
            layers.BatchNormalization(),  # Normalize activations for stable training
            layers.LeakyReLU(),          # Non-linear activation
            layers.Reshape((7, 7, 256)),  # Reshape to initial feature map
            
            # First upsampling block: 7x7 -> 14x14
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Second upsampling block: 14x14 -> 28x28
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Third upsampling block: 28x28 -> 56x56
            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Final layer to generate RGB image
            # Output is normalized to [-1, 1] using tanh activation
            layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', 
                                 use_bias=False, activation='tanh')
        ])
        
    def call(self, inputs):
        """
        Forward pass of the generator.
        
        Args:
            inputs: Input tensor of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, 56, 56, 3)
        """
        return self.model(inputs)
    
    def generate_images(self, num_images):
        """
        Generate a batch of synthetic images from random noise.
        
        Args:
            num_images (int): Number of images to generate
            
        Returns:
            Tensor of generated images with shape (num_images, 56, 56, 3)
        """
        # Generate random noise vectors
        random_noise = tf.random.normal([num_images, self.latent_dim])
        # Generate images from noise
        generated_images = self(random_noise, training=False)
        return generated_images 