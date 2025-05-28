import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
import json
from datetime import datetime

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
DROPOUT_RATE = 0.7
L2_LAMBDA = 0.003
CONFIDENCE_THRESHOLD = 0.7

def create_resnet_model(num_classes):
    print("\nCreating ResNet50 model architecture...")
    
    # Use mixed precision for faster GPU training (if available)
    if tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Using mixed precision policy for faster GPU training")
    
    # Load pre-trained ResNet50
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("ResNet50 architecture created successfully")
    model.summary()
    return model, base_model

def train_resnet_model():
    print("\n=== Starting ResNet50 Model Training Process ===")
    
    # Try multiple potential dataset locations
    possible_paths = [
        "dataset",
        "Augmented Dataset",
        "./Dragon Fruit Maturity Detection Dataset/Augmented Dataset",
        "../Dragon Fruit Maturity Detection Dataset/Augmented Dataset",
        "Dragon Fruit Maturity Detection Dataset/Augmented Dataset",
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = path
            print(f"Found valid dataset path: {data_dir}")
            break
    
    if data_dir is None:
        raise ValueError("Could not find dataset directory")
    
    # Load and preprocess data (reuse the same function from train.py)
    from train import load_and_preprocess_data
    train_ds, test_ds, class_names, class_weights = load_and_preprocess_data(data_dir, batch_size=BATCH_SIZE)
    
    # Create and compile model
    print("\nCreating and compiling ResNet50 model...")
    model, base_model = create_resnet_model(len(class_names))
    
    # Learning rate scheduler
    initial_learning_rate = 0.0001  # Lower learning rate for transfer learning
    decay_steps = 1000
    decay_rate = 0.95
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Create timestamp for unique log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'./logs/resnet_{timestamp}',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    
    callbacks = [early_stopping, reduce_lr, tensorboard_callback]
    
    # First phase: Train only the top layers
    print("\nPhase 1: Training top layers...")
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Second phase: Fine-tune the ResNet layers
    print("\nPhase 2: Fine-tuning ResNet layers...")
    base_model.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Combine histories
    history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    # Save model and info
    model_save_path = 'resnet_fruit_maturity_model.h5'
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save model info
    model_info = {
        'class_names': class_names,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'training_parameters': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS * 2,  # Total epochs from both phases
            'dropout_rate': DROPOUT_RATE,
            'l2_lambda': L2_LAMBDA,
            'initial_learning_rate': initial_learning_rate,
            'decay_steps': decay_steps,
            'decay_rate': decay_rate,
            'class_weights': class_weights
        }
    }
    
    with open('resnet_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('ResNet50 Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('ResNet50 Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet_training_results.png')
    
    # Evaluate model
    print("\nEvaluating ResNet50 model...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}")
    
    # Generate predictions and calculate metrics
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        predicted_classes = tf.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes.numpy())
    
    # Calculate metrics
    from train import calculate_metrics, print_metrics_table
    metrics_dict = calculate_metrics(y_true, y_pred, class_names)
    print_metrics_table(metrics_dict, class_names)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(columns=[
        'Class', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision',
        'Recall', 'F1 Score', 'Specificity'
    ])
    
    for class_name, metrics in metrics_dict.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            'Class': class_name,
            'TP': metrics['TP'],
            'TN': metrics['TN'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1 Score': metrics['F1 Score'],
            'Specificity': metrics['Specificity']
        }])], ignore_index=True)
    
    metrics_df.to_csv('resnet_detailed_metrics.csv', index=False)
    print("\nDetailed metrics saved to resnet_detailed_metrics.csv")
    
    return model, class_names, metrics_dict

def compare_models():
    """Compare AlexNet and ResNet50 models"""
    print("\n=== Model Comparison ===")
    
    # Load metrics from both models
    alexnet_metrics = pd.read_csv('detailed_metrics.csv')
    resnet_metrics = pd.read_csv('resnet_detailed_metrics.csv')
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame()
    comparison_df['Class'] = alexnet_metrics['Class']
    
    # Compare each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
    for metric in metrics:
        comparison_df[f'AlexNet_{metric}'] = alexnet_metrics[metric]
        comparison_df[f'ResNet_{metric}'] = resnet_metrics[metric]
        comparison_df[f'{metric}_Difference'] = resnet_metrics[metric] - alexnet_metrics[metric]
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nModel comparison saved to model_comparison.csv")
    
    # Print comparison table
    print("\nModel Comparison Summary:")
    print("-" * 100)
    print(f"{'Class':<25} {'Metric':<10} {'AlexNet':<10} {'ResNet':<10} {'Difference':<10}")
    print("-" * 100)
    
    for _, row in comparison_df.iterrows():
        for metric in metrics:
            print(f"{row['Class']:<25} {metric:<10} "
                  f"{row[f'AlexNet_{metric}']:<10.4f} "
                  f"{row[f'ResNet_{metric}']:<10.4f} "
                  f"{row[f'{metric}_Difference']:<10.4f}")
        print("-" * 100)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        x = np.arange(len(comparison_df['Class']))
        width = 0.35
        
        plt.bar(x - width/2, comparison_df[f'AlexNet_{metric}'], width, label='AlexNet')
        plt.bar(x + width/2, comparison_df[f'ResNet_{metric}'], width, label='ResNet50')
        
        plt.xlabel('Class')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison')
        plt.xticks(x, comparison_df['Class'], rotation=45)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nComparison plot saved to model_comparison.png")

def main():
    print("Starting ResNet50 model training...")
    
    try:
        # Train the ResNet model
        model, class_names, metrics_dict = train_resnet_model()
        
        # Compare with AlexNet
        compare_models()
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main() 