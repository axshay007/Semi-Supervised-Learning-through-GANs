import tensorflow as tf
from .generator import Generator
from .discriminator import Discriminator

class MelanomaSemiSupervisedGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
    @tf.function
    def train_step_gan(self, real_images):
        """Train both generator and discriminator on unlabeled data."""
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator
        random_noise = tf.random.normal([batch_size, self.latent_dim])
        generated_images = self.generator(random_noise, training=True)
        d_loss = self.discriminator.train_step_gan(real_images, generated_images, self.discriminator_optimizer)
        
        # Train generator
        with tf.GradientTape() as tape:
            random_noise = tf.random.normal([batch_size, self.latent_dim])
            generated_images = self.generator(random_noise, training=True)
            fake_output, _ = self.discriminator(generated_images, training=False)
            g_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)
            
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss
    
    def train_step_supervised(self, labeled_images, labels):
        """Train discriminator on labeled data for melanoma classification."""
        return self.discriminator.train_step_melanoma(labeled_images, labels, self.discriminator_optimizer)
    
    def train(self, labeled_dataset, unlabeled_dataset, epochs=100):
        """Full training loop combining supervised and unsupervised learning."""
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
            
            # Print metrics
            avg_d_loss = sum(d for d, _ in gan_losses) / len(gan_losses)
            avg_g_loss = sum(g for _, g in gan_losses) / len(gan_losses)
            avg_supervised_loss = sum(supervised_losses) / len(supervised_losses)
            
            print(f"Discriminator Loss: {avg_d_loss:.4f}")
            print(f"Generator Loss: {avg_g_loss:.4f}")
            print(f"Supervised Loss: {avg_supervised_loss:.4f}")
    
    def save_models(self, save_dir):
        """Save both generator and discriminator models."""
        self.generator.save(f"{save_dir}/generator")
        self.discriminator.save(f"{save_dir}/discriminator")
    
    def load_models(self, save_dir):
        """Load both generator and discriminator models."""
        self.generator = tf.keras.models.load_model(f"{save_dir}/generator")
        self.discriminator = tf.keras.models.load_model(f"{save_dir}/discriminator") 