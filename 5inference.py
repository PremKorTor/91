import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
import pickle

# Configuration
MODEL_DIR = 'examples'
DATA_PATH = 'data/test.csv'

def clean_data(df):
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    return df

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    ix = np.argmax(f1_scores)
    return thresholds[ix]

def plot_custom_confusion_matrix(cm, accuracy, save_path):
    """
    Generates a confusion matrix exactly matching the user's requested style:
    - Blue heatmap
    - Side text box with statistics
    """
    tn, fp, fn, tp = cm.ravel()
    
    # Create the text string for the box
    stats_text = (
        f"True Negatives: {tn}\n"
        f"False Positives: {fp}\n"
        f"False Negatives: {fn}\n"
        f"True Positives: {tp}\n\n"
        f"Accuracy: {accuracy:.4f}"
    )

    # Create figure with extra width for the text box
    plt.figure(figsize=(10, 6))
    
    # Plot heatmap (using only 80% of width to leave room for text)
    ax = plt.gca()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    
    # Styling
    plt.title('Confusion Matrix - Ass5 Classification Model', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add the text box on the right side
    # bbox props creates the beige box style
    plt.gcf().text(0.82, 0.6, stats_text, fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
    
    # Adjust layout to prevent cutting off the text
    plt.subplots_adjust(left=0.1, right=0.75)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved custom matrix to: {save_path}")

def main():
    print("Loading resources...")
    model_path = os.path.join(MODEL_DIR, 'best_model.h5')
    if not os.path.exists(model_path):
        print("Model not found. Run training first.")
        return

    model = keras.models.load_model(model_path)
    
    with open(os.path.join(MODEL_DIR, 'artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)
        
    preprocessor = artifacts['preprocessor']
    X_val, y_val = artifacts['val_data']

    # Tuning
    print("Tuning threshold...")
    val_probs = model.predict(X_val)
    optimal_threshold = find_optimal_threshold(y_val, val_probs)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Process Test Data
    print("Processing Test Data...")
    df_test = pd.read_csv(DATA_PATH)
    
    if 'ProdTaken' in df_test.columns:
        y_test_true = df_test['ProdTaken']
        X_test = df_test.drop('ProdTaken', axis=1)
        has_labels = True
    else:
        X_test = df_test
        has_labels = False

    X_test = clean_data(X_test)
    X_test_processed = preprocessor.transform(X_test)

    # Predict
    test_probs = model.predict(X_test_processed)
    y_test_pred = (test_probs >= optimal_threshold).astype(int).flatten()

    if has_labels:
        print("\n" + "="*30)
        print(f"FINAL EVALUATION (Threshold: {optimal_threshold:.4f})")
        print("="*30)
        
        report = classification_report(y_test_true, y_test_pred)
        print(report)
        
        # Metrics
        cm = confusion_matrix(y_test_true, y_test_pred)
        acc = accuracy_score(y_test_true, y_test_pred)

        # --- OUTPUT 1: Standard Blue Matrix with Stats Box ---
        plot_custom_confusion_matrix(cm, acc, os.path.join(MODEL_DIR, 'confusion_matrix_standard.png'))

        # --- OUTPUT 2: Normalized Green Matrix ---
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', 
                    xticklabels=['0', '1'], 
                    yticklabels=['0', '1'])
        plt.title('Normalized Confusion Matrix - Ass5 Classification Model', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_normalized.png'))
        plt.close()
        print(f"Saved normalized matrix to: {os.path.join(MODEL_DIR, 'confusion_matrix_normalized.png')}")
        
        # Save Predictions CSV
        results_df = df_test.copy()
        results_df['Predicted_Prob'] = test_probs
        results_df['Predicted_Class'] = y_test_pred
        results_df.to_csv(os.path.join(MODEL_DIR, 'test_predictions.csv'), index=False)
        print("\nPredictions saved.")

if __name__ == "__main__":
    main()