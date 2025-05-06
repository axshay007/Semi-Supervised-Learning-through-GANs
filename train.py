import os
import tensorflow as tf
from models.gan import MelanomaSemiSupervisedGAN
from utils.ham10000_loader import (
    create_data_generators,
    create_test_generator,
    normalize_images
)
from utils.visualization import save_training_progress
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Melanoma Semi-Supervised GAN')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size_labeled', type=int, default=32,
                        help='Batch size for labeled data')
    parser.add_argument('--batch_size_unlabeled', type=int, default=128,
                        help='Batch size for unlabeled data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of the latent space')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints and visualizations')
    parser.add_argument('--image_size', type=int, default=56,
                        help='Size of the input images')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create data generators
    labeled_generator, unlabeled_generator = create_data_generators(
        train_dir=os.path.join(args.data_dir, 'labeled'),
        unlabeled_dir=os.path.join(args.data_dir, 'unlabeled'),
        target_size=(args.image_size, args.image_size),
        mb_size_labeled=args.batch_size_labeled,
        mb_size_unlabeled=args.batch_size_unlabeled
    )
    
    # Create test generator
    test_generator = create_test_generator(
        test_dir=os.path.join(args.data_dir, 'test'),
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size_labeled
    )
    
    # Initialize model
    model = MelanomaSemiSupervisedGAN(latent_dim=args.latent_dim)
    
    # Training metrics history
    metrics_history = {
        'd_loss': [],
        'g_loss': [],
        'supervised_loss': []
    }
    
    # Training loop
    try:
        print("Starting training...")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train on unlabeled data (GAN training)
            gan_losses = []
            for _ in range(len(unlabeled_generator)):
                images = next(unlabeled_generator)
                # Convert images from [0, 1] to [-1, 1] range
                images = normalize_images(images)
                d_loss, g_loss = model.train_step_gan(images)
                gan_losses.append((float(d_loss), float(g_loss)))
            
            # Train on labeled data (supervised learning)
            supervised_losses = []
            for _ in range(len(labeled_generator)):
                images, labels = next(labeled_generator)
                # Convert images from [0, 1] to [-1, 1] range
                images = normalize_images(images)
                loss = model.train_step_supervised(images, labels)
                supervised_losses.append(float(loss))
            
            # Update metrics history
            avg_d_loss = sum(d for d, _ in gan_losses) / len(gan_losses)
            avg_g_loss = sum(g for _, g in gan_losses) / len(gan_losses)
            avg_supervised_loss = sum(supervised_losses) / len(supervised_losses)
            
            metrics_history['d_loss'].append(avg_d_loss)
            metrics_history['g_loss'].append(avg_g_loss)
            metrics_history['supervised_loss'].append(avg_supervised_loss)
            
            print(f"Discriminator Loss: {avg_d_loss:.4f}")
            print(f"Generator Loss: {avg_g_loss:.4f}")
            print(f"Supervised Loss: {avg_supervised_loss:.4f}")
            
            # Save progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_training_progress(epoch, model.generator, metrics_history, args.save_dir)
                model.save_models(args.save_dir)
                
                # Evaluate on test set
                test_metrics = model.discriminator.evaluate(test_generator)
                print(f"\nTest accuracy: {test_metrics[1]:.4f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
    
    # Save final model
    model.save_models(args.save_dir)
    print(f"Training completed. Models saved in {args.save_dir}")

if __name__ == '__main__':
    main() 