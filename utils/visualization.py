import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_generated_images(generator, num_examples=16, latent_dim=100):
    """Plot a grid of generated images."""
    noise = tf.random.normal([num_examples, latent_dim])
    generated_images = generator(noise, training=False)
    
    # Rescale images from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2.0
    
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(num_examples):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_metrics(metrics_history):
    """Plot training metrics over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot GAN losses
    ax1.plot(metrics_history['d_loss'], label='Discriminator Loss')
    ax1.plot(metrics_history['g_loss'], label='Generator Loss')
    ax1.set_title('GAN Training Progress')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot supervised loss
    ax2.plot(metrics_history['supervised_loss'], label='Supervised Loss')
    ax2.set_title('Supervised Training Progress')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, classes=['Benign', 'Malignant']):
    """Plot confusion matrix for melanoma classification."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    return fig

def save_training_progress(epoch, generator, metrics_history, save_dir):
    """Save training progress visualizations."""
    # Create generated images plot
    gen_fig = plot_generated_images(generator)
    gen_fig.savefig(f'{save_dir}/generated_images_epoch_{epoch}.png')
    plt.close(gen_fig)
    
    # Create metrics plot
    metrics_fig = plot_training_metrics(metrics_history)
    metrics_fig.savefig(f'{save_dir}/training_metrics_epoch_{epoch}.png')
    plt.close(metrics_fig) 