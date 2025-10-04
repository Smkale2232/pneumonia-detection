# main.py - COMPLETE FIXED VERSION
import tensorflow as tf
import yaml
import os
import sys
import numpy as np

# Add src to path
sys.path.append('src')

from data_loader import DataLoader
from model import create_pneumonia_model
from train import Trainer
from evaluate import Evaluator
from grad_cam import GradCAM
from utils import plot_training_history
from gpu_optimizer import optimize_gpu_for_1650

def check_dataset_exists():
    """Check if dataset exists and has images"""
    required_dirs = [
        'data/chest_xray/train/NORMAL',
        'data/chest_xray/train/PNEUMONIA',
        'data/chest_xray/test/NORMAL',
        'data/chest_xray/test/PNEUMONIA'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            return False, f"Missing directory: {dir_path}"
        
        # Check if directory has images
        images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        if len(images) == 0:
            return False, f"No images found in: {dir_path}"
    
    return True, "Dataset OK"

def main():
    print("=== Pneumonia Detection - GTX 1650 Optimized ===")
    
    # Check dataset first
    dataset_ok, dataset_message = check_dataset_exists()
    if not dataset_ok:
        print(f"❌ Dataset issue: {dataset_message}")
        print("\nPlease:")
        print("1. Download dataset from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
        print("2. Extract to: data/chest_xray/")
        print("3. Ensure structure is: data/chest_xray/train/NORMAL/, data/chest_xray/train/PNEUMONIA/, etc.")
        print("\nYou can run: python download_dataset.py for instructions")
        return
    
    # Setup GPU
    print("Step 0: GPU Optimization")
    gpu_optimized = optimize_gpu_for_1650()
    
    # Load configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Config file not found. Creating default config...")
        # Create default config
        config = {
            'model': {
                'base_model': 'MobileNetV2',
                'img_size': [224, 224],
                'batch_size': 16,
                'learning_rate': 0.001,
                'fine_tune_learning_rate': 0.0001
            },
            'training': {
                'epochs': 25,
                'fine_tune_epochs': 15,
                'patience': 8,
                'unfreeze_layers': 40
            },
            'data': {
                'train_dir': 'data/chest_xray/train',
                'val_dir': 'data/chest_xray/val',
                'test_dir': 'data/chest_xray/test'
            },
            'paths': {
                'model_save': 'models/best_pneumonia_model.h5',
                'logs': 'logs/',
                'results': 'results/'
            }
        }
        # Create config directory if it doesn't exist
        os.makedirs('config', exist_ok=True)
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("✓ Created default config file")
    
    print("Step 1: Data Setup")
    data_loader = DataLoader(config)
    train_ds, val_ds, test_ds = data_loader.load_datasets()
    class_weights = data_loader.calculate_class_weights(train_ds)
    train_ds, val_ds, test_ds = data_loader.optimize_datasets(train_ds, val_ds, test_ds)
    
    print(f"Class weights: {class_weights}")
    
    print("Step 2 & 3: Model Creation")
    model, base_model = create_pneumonia_model(config)
    
    print("Step 4: Training")
    trainer = Trainer(model, config)
    
    # Check if we have a saved model to load
    if os.path.exists(config['paths']['model_save']):
        print("Found existing model. Do you want to:")
        print("1. Train new model")
        print("2. Load existing model and continue training")
        print("3. Load existing model and skip to evaluation")
        
        # For now, we'll train new model. You can modify this interactive part later.
        choice = "1"
    else:
        choice = "1"
    
    if choice == "1":
        # Train new model
        history = trainer.train(train_ds, val_ds, class_weights)
        
        # Check if we should fine-tune
        if 'val_auc' in history.history:
            best_val_auc = max(history.history['val_auc'])
            print(f"Best validation AUC from initial training: {best_val_auc:.4f}")
            
            # Fine-tune only if performance can be improved
            if best_val_auc < 0.95:  # Threshold for fine-tuning
                print("Starting fine-tuning...")
                try:
                    history_fine = trainer.fine_tune(
                        base_model, train_ds, val_ds, class_weights, 
                        initial_epoch=len(history.history['loss'])
                    )
                    final_history = history_fine
                except Exception as e:
                    print(f"Fine-tuning failed: {e}")
                    print("Continuing with initial training model...")
                    final_history = history
            else:
                print("Skipping fine-tuning - model already performing excellently!")
                final_history = history
        else:
            print("No AUC metric available, skipping fine-tuning")
            final_history = history
            
    elif choice == "2":
        # Load and continue training
        print("Loading existing model...")
        model = tf.keras.models.load_model(config['paths']['model_save'])
        trainer = Trainer(model, config)
        history = trainer.train(train_ds, val_ds, class_weights)
        final_history = history
        
    else:  # choice == "3"
        # Load and skip to evaluation
        print("Loading existing model for evaluation...")
        model = tf.keras.models.load_model(config['paths']['model_save'])
        final_history = None
    
    # Load the best model for evaluation (whether from initial training or fine-tuning)
    if os.path.exists(config['paths']['model_save']):
        print("Loading best model for evaluation...")
        best_model = tf.keras.models.load_model(config['paths']['model_save'])
    else:
        print("Using current model for evaluation...")
        best_model = model
    
    print("Step 5: Evaluation")
    evaluator = Evaluator(best_model, config)
    
    try:
        y_true, y_pred, y_pred_binary = evaluator.evaluate_model(test_ds)
    except Exception as e:
        print(f"Evaluation error: {e}")
        print("Using alternative evaluation method...")
        
        # Manual evaluation as fallback
        y_pred = best_model.predict(test_ds)
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        
        # Calculate metrics manually
        from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        
        print(f"\n=== Manual Test Set Evaluation ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        
        print(f"\n=== Classification Report ===")
        print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Pneumonia']))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        os.makedirs('results/training_plots', exist_ok=True)
        plt.savefig('results/training_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("Step 6: Grad-CAM Visualization")
    try:
        grad_cam = GradCAM(best_model, config)
        grad_cam.visualize_for_batch(test_ds)
    except Exception as e:
        print(f"Grad-CAM visualization failed: {e}")
        print("Skipping Grad-CAM for now...")
    
    print("Step 7: Training History")
    if final_history is not None:
        try:
            plot_training_history(history, final_history if 'history_fine' in locals() else None, 
                                 save_path='results/training_plots/training_history.png')
        except Exception as e:
            print(f"Training history plot failed: {e}")
    else:
        print("No training history available to plot")
    
    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETE!")
    print("="*50)
    print("\nResults saved in:")
    print("📊 Metrics: results/metrics/")
    print("📈 Plots: results/training_plots/")
    print("🔥 Grad-CAM: results/grad_cam_visualizations/")
    print("💾 Model: models/best_pneumonia_model.h5")
    
    # Print final recommendations
    if 'y_true' in locals() and 'y_pred' in locals():
        from sklearn.metrics import accuracy_score
        final_accuracy = accuracy_score(y_true, y_pred_binary)
        if final_accuracy > 0.90:
            print("\n✅ Excellent model performance!")
        elif final_accuracy > 0.80:
            print("\n👍 Good model performance!")
        else:
            print("\n⚠️  Model performance needs improvement. Consider:")
            print("   - More data augmentation")
            print("   - Trying different base models")
            print("   - Adjusting hyperparameters")

if __name__ == "__main__":
    main()