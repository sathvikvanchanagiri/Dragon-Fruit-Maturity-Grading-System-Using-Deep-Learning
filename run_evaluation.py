import tensorflow as tf
from evaluate_model import generate_metrics_report
from train import load_and_preprocess_data
import os

def main():
    # Define paths
    model_path = 'fruit_maturity_model.h5'  # Updated path
    model_info_path = 'model_info.json'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure the model has been trained and saved correctly.")
        return
    
    if not os.path.exists(model_info_path):
        print(f"Error: Model info file not found at {model_info_path}")
        return
    
    try:
        # Load the model
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Load the test dataset
        print("Loading test dataset...")
        data_dir = "../Dragon Fruit Maturity Detection Dataset/Augmented Dataset"
        _, test_ds, _ = load_and_preprocess_data(data_dir, batch_size=32)
        print("Test dataset loaded successfully")
        
        # Generate metrics report
        metrics_dict, cm, overall_metrics = generate_metrics_report(
            model=model,
            test_ds=test_ds,
            output_dir='model_output'
        )
        
        print("\nDetailed evaluation completed. Check the following files in the model_output directory:")
        print("1. confusion_matrix.png - Visual representation of the confusion matrix")
        print("2. confusion_metrics.csv - Detailed TP, TN, FP, FN for each class")
        print("3. performance_metrics.csv - Accuracy, Precision, Recall, F1 Score, Specificity")
        print("4. overall_metrics.json - Overall classification metrics")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        print("Please check the error message above and ensure all required files exist.")

if __name__ == "__main__":
    main() 