import tensorflow as tf
from tensorflow.keras import layers, Model

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Shared convolutional layers for both tasks
        self.shared_layers = tf.keras.Sequential([
            # 56x56x3 input
            layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # 28x28x32
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # 14x14x64
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # 7x7x128
            layers.Flatten()
        ])
        
        # Real/Fake classification branch
        self.gan_classifier = tf.keras.Sequential([
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Melanoma classification branch
        self.melanoma_classifier = tf.keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.4),
            layers.Dense(2, activation='softmax')  # 2 classes: benign and malignant
        ])
    
    def call(self, inputs, training=True):
        features = self.shared_layers(inputs, training=training)
        gan_output = self.gan_classifier(features)
        melanoma_output = self.melanoma_classifier(features)
        return gan_output, melanoma_output
    
    @tf.function
    def train_step_gan(self, real_images, fake_images, optimizer):
        """Training step for GAN classification."""
        with tf.GradientTape() as tape:
            # Real images should be classified as 1
            real_gan_output, _ = self(real_images, training=True)
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_gan_output), real_gan_output)
            
            # Fake images should be classified as 0
            fake_gan_output, _ = self(fake_images, training=True)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_gan_output), fake_gan_output)
            
            total_loss = real_loss + fake_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss
    
    @tf.function
    def train_step_melanoma(self, images, labels, optimizer):
        """Training step for melanoma classification."""
        with tf.GradientTape() as tape:
            _, melanoma_output = self(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, melanoma_output)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss 