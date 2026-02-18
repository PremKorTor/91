import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# Configuration
DATA_PATH = 'data/train_validation.csv'
MODEL_SAVE_DIR = 'examples'
EPOCHS = 100
BATCH_SIZE = 32

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def clean_data(df):
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    return df

def get_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    return preprocessor

def build_model(input_dim):
    # CHANGED BACK: Smaller model to prevent overfitting and improve F1
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Reduced form 256 to 64
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Reduced from 128 to 32
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def main():
    print("Loading and cleaning data...")
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)

    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Preprocessing...")
    preprocessor = get_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Class Weights
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights_vals))
    print(f"Computed Class Weights: {class_weights_dict}")

    artifacts = {
        'preprocessor': preprocessor,
        'val_data': (X_val_processed, y_val)
    }
    with open(os.path.join(MODEL_SAVE_DIR, 'artifacts.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)

    model = build_model(X_train_processed.shape[1])
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.h5')

    my_callbacks = [
        callbacks.ModelCheckpoint(model_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        X_train_processed, y_train, 
        validation_data=(X_val_processed, y_val),
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=my_callbacks,
        class_weight=class_weights_dict
    )

    # Training History Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_history.png'))
    print("Training Complete.")

if __name__ == "__main__":
    main()