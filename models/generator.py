import tensorflow as tf
from tensorflow.keras import layers, Model

class Generator(Model):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Define the generator architecture
        self.model = tf.keras.Sequential([
            # Starting with 7x7x256
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            
            # Upsampling to 14x14
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Upsampling to 28x28
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Upsampling to 56x56
            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Final output layer 56x56x3
            layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
        ])
        
    def call(self, inputs):
        return self.model(inputs)
    
    def generate_images(self, num_images):
        """Generate a batch of images from random noise."""
        random_noise = tf.random.normal([num_images, self.latent_dim])
        generated_images = self(random_noise, training=False)
        return generated_images 