import os
import tensorflow as tf
import numpy as np
from models.gan import MelanomaSemiSupervisedGAN
from utils.data_loader import create_labeled_dataset
from utils.visualization import plot_confusion_matrix, plot_generated_images
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Melanoma Semi-Supervised GAN')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing test data')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                        help='Directory containing trained models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def evaluate_model(model, test_dataset):
    """Evaluate model performance on test dataset."""
    all_predictions = []
    all_labels = []
    
    for images, labels in test_dataset:
        _, melanoma_predictions = model.discriminator(images, training=False)
        predictions = tf.argmax(melanoma_predictions, axis=1)
        
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    test_dataset = create_labeled_dataset(args.data_dir, args.batch_size)
    
    # Initialize and load trained model
    model = MelanomaSemiSupervisedGAN()
    model.load_models(args.model_dir)
    
    # Evaluate model
    predictions, true_labels = evaluate_model(model, test_dataset)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, accuracy_score
    accuracy = accuracy_score(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, 
                                      target_names=['Benign', 'Malignant'])
    
    # Save results
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Plot and save confusion matrix
    cm_fig = plot_confusion_matrix(true_labels, predictions)
    cm_fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Generate and save sample images
    gen_fig = plot_generated_images(model.generator)
    gen_fig.savefig(os.path.join(args.output_dir, 'generated_samples.png'))
    
    print(f"Evaluation completed. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main() 