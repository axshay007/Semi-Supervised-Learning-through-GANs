import os
import tensorflow as tf
import numpy as np
from models.gan import MelanomaSemiSupervisedGAN
from utils.ham10000_loader import create_test_generator, normalize_images
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
    parser.add_argument('--image_size', type=int, default=56,
                        help='Size of the input images')
    return parser.parse_args()

def evaluate_model(model, test_generator):
    """
    Evaluate model performance on test dataset.
    
    Args:
        model: Trained SGAN model
        test_generator: Generator for test data
        
    Returns:
        tuple: (predictions, true_labels, accuracy, precision, recall, f1_score)
    """
    all_predictions = []
    all_labels = []
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions for all test images
    for _ in range(len(test_generator)):
        images, labels = next(test_generator)
        images = normalize_images(images)  # Convert to [-1, 1] range
        _, melanoma_predictions = model.discriminator(images, training=False)
        predictions = tf.argmax(melanoma_predictions, axis=1)
        
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    return np.array(all_predictions), np.array(all_labels), accuracy, precision, recall, f1_score

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    test_generator = create_test_generator(
        test_dir=os.path.join(args.data_dir, 'test'),
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size
    )
    
    # Initialize and load trained model
    model = MelanomaSemiSupervisedGAN()
    model.load_models(args.model_dir)
    
    # Evaluate model
    predictions, true_labels, accuracy, precision, recall, f1_score = evaluate_model(
        model, test_generator
    )
    
    # Save results
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n\n")
        
        # Add dataset information
        f.write("Dataset Information\n")
        f.write("==================\n")
        f.write(f"Total test images: {len(true_labels)}\n")
        f.write(f"Benign images: {sum(true_labels == 0)}\n")
        f.write(f"Malignant images: {sum(true_labels == 1)}\n")
    
    # Plot and save confusion matrix
    cm_fig = plot_confusion_matrix(true_labels, predictions)
    cm_fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Generate and save sample images
    gen_fig = plot_generated_images(model.generator)
    gen_fig.savefig(os.path.join(args.output_dir, 'generated_samples.png'))
    
    print(f"Evaluation completed. Results saved in {args.output_dir}")
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == '__main__':
    main() 