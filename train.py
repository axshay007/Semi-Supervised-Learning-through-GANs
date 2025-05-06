import os
import tensorflow as tf
from models.gan import MelanomaSemiSupervisedGAN
from utils.data_loader import (
    create_labeled_dataset,
    create_unlabeled_dataset,
    create_augmented_dataset
)
from utils.visualization import save_training_progress
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Melanoma Semi-Supervised GAN')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing labeled and unlabeled data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of the latent space')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints and visualizations')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets
    labeled_dataset = create_labeled_dataset(args.data_dir, args.batch_size)
    unlabeled_dataset = create_unlabeled_dataset(args.data_dir, args.batch_size)
    
    # Apply data augmentation
    labeled_dataset = create_augmented_dataset(labeled_dataset)
    unlabeled_dataset = create_augmented_dataset(unlabeled_dataset)
    
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
        model.train(labeled_dataset, unlabeled_dataset, epochs=args.epochs)
        
        # Save training progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_training_progress(epoch, model.generator, metrics_history, args.save_dir)
            model.save_models(args.save_dir)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
    
    # Save final model
    model.save_models(args.save_dir)
    print(f"Training completed. Models saved in {args.save_dir}")

if __name__ == '__main__':
    main() 