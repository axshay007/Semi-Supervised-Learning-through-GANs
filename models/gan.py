import tensorflow as tf
from .generator import Generator
from .discriminator import Discriminator

class MelanomaSemiSupervisedGAN:
    """
    Semi-Supervised GAN for Melanoma Detection
    
    This class implements a semi-supervised learning approach using GANs for melanoma detection.
    It combines:
    1. Unsupervised learning through GAN training on unlabeled images
    2. Supervised learning for melanoma classification on labeled images
    
    The architecture consists of:
    - A Generator that creates synthetic skin lesion images
    - A Discriminator that performs both real/fake and melanoma classification
    
    The training process alternates between:
    - GAN training: Generator creates images, Discriminator learns to distinguish real/fake
    - Supervised training: Discriminator learns to classify melanomas using labeled data
    """
    
    def __init__(self, latent_dim=100):
        """
        Initialize the Semi-Supervised GAN model.
        
        Args:
            latent_dim (int): Dimension of the generator's input noise vector
        """
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        # Initialize optimizers with beta1=0.5 for stable GAN training
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
    @tf.function
    def train_step_gan(self, real_images):
        """
        Perform one training step of GAN training.
        
        This method:
        1. Generates fake images from random noise
        2. Updates the discriminator to better distinguish real/fake images
        3. Updates the generator to create more realistic images
        
        Args:
            real_images: Batch of real skin lesion images
            
        Returns:
            tuple: (discriminator_loss, generator_loss)
        """
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator
        random_noise = tf.random.normal([batch_size, self.latent_dim])
        generated_images = self.generator(random_noise, training=True)
        d_loss = self.discriminator.train_step_gan(
            real_images, generated_images, self.discriminator_optimizer)
        
        # Train generator
        with tf.GradientTape() as tape:
            random_noise = tf.random.normal([batch_size, self.latent_dim])
            generated_images = self.generator(random_noise, training=True)
            fake_output, _ = self.discriminator(generated_images, training=False)
            # Generator tries to fool discriminator into classifying fake images as real
            g_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output)
            
        # Update generator weights
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss
    
    def train_step_supervised(self, labeled_images, labels):
        """
        Perform one step of supervised training for melanoma classification.
        
        Args:
            labeled_images: Batch of labeled skin lesion images
            labels: Corresponding melanoma/benign labels
            
        Returns:
            float: Classification loss
        """
        return self.discriminator.train_step_melanoma(
            labeled_images, labels, self.discriminator_optimizer)
    
    def train(self, labeled_dataset, unlabeled_dataset, epochs=100):
        """
        Full training loop combining supervised and unsupervised learning.
        
        For each epoch:
        1. Train the GAN on unlabeled data
        2. Train the discriminator on labeled data for melanoma classification
        
        Args:
            labeled_dataset: TensorFlow dataset of labeled images
            unlabeled_dataset: TensorFlow dataset of unlabeled images
            epochs (int): Number of training epochs
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train on unlabeled data (GAN training)
            gan_losses = []
            for batch in unlabeled_dataset:
                d_loss, g_loss = self.train_step_gan(batch)
                gan_losses.append((float(d_loss), float(g_loss)))
            
            # Train on labeled data (supervised learning)
            supervised_losses = []
            for images, labels in labeled_dataset:
                loss = self.train_step_supervised(images, labels)
                supervised_losses.append(float(loss))
            
            # Calculate and print metrics
            avg_d_loss = sum(d for d, _ in gan_losses) / len(gan_losses)
            avg_g_loss = sum(g for _, g in gan_losses) / len(gan_losses)
            avg_supervised_loss = sum(supervised_losses) / len(supervised_losses)
            
            print(f"Discriminator Loss: {avg_d_loss:.4f}")
            print(f"Generator Loss: {avg_g_loss:.4f}")
            print(f"Supervised Loss: {avg_supervised_loss:.4f}")
    
    def save_models(self, save_dir):
        """
        Save both generator and discriminator models.
        
        Args:
            save_dir (str): Directory to save the models
        """
        self.generator.save(f"{save_dir}/generator")
        self.discriminator.save(f"{save_dir}/discriminator")
    
    def load_models(self, save_dir):
        """
        Load both generator and discriminator models.
        
        Args:
            save_dir (str): Directory containing the saved models
        """
        self.generator = tf.keras.models.load_model(f"{save_dir}/generator")
        self.discriminator = tf.keras.models.load_model(f"{save_dir}/discriminator") 