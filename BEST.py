#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from scipy.stats import entropy, skew, kurtosis
from collections import Counter
import zlib
from typing import List, Tuple
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility functions
def hex_to_bytes(hex_string: str) -> bytes:
    return bytes.fromhex(hex_string)

def bytes_to_int_array(byte_data: bytes) -> np.ndarray:
    return np.frombuffer(byte_data, dtype=np.uint8)

def calculate_entropy(data: np.ndarray) -> float:
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return entropy(probabilities)

def calculate_byte_frequency(data: np.ndarray) -> np.ndarray:
    return np.bincount(data, minlength=256) / len(data)

def autocorrelation(data: np.ndarray, lag: int = 1) -> float:
    return np.corrcoef(data[:-lag], data[lag:])[0, 1]

def kolmogorov_complexity(data: bytes) -> float:
    return len(zlib.compress(data)) / len(data)

def n_gram_analysis(data: np.ndarray, n: int) -> float:
    ngrams = [''.join(map(str, data[i:i+n])) for i in range(len(data)-n+1)]
    return entropy(list(Counter(ngrams).values()))

def process_hex_data(hex_string):
    hex_string = ''.join(hex_string.split())
    hex_string = ''.join(c for c in hex_string if c in '0123456789ABCDEFabcdef')
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string
    return hex_string

def extract_features(byte_data: bytes, n_features: int = 258) -> np.ndarray:
    int_data = bytes_to_int_array(byte_data)
    
    basic_features = [
        calculate_entropy(int_data),
        np.mean(int_data),
        np.std(int_data),
        len(int_data),
        np.median(int_data),
        skew(int_data),
        kurtosis(int_data),
        np.max(int_data),
        np.min(int_data),
        np.ptp(int_data)  # Peak-to-peak (range) of data
    ]
    
    byte_freq = calculate_byte_frequency(int_data)
    autocorr = [autocorrelation(int_data, lag) for lag in [1, 2, 4, 8]]
    kol_complexity = kolmogorov_complexity(byte_data)
    ngram_features = [n_gram_analysis(int_data, n) for n in [2, 3, 4]]
    
    fft_features = np.abs(np.fft.fft(int_data))[:5]  # First 5 FFT coefficients
    diff_features = np.diff(int_data)
    diff_stats = [np.mean(diff_features), np.std(diff_features), skew(diff_features), kurtosis(diff_features)]
    
    all_features = np.concatenate([basic_features, byte_freq, autocorr, [kol_complexity], 
                                   ngram_features, fft_features, diff_stats])
    
    # Ensure we always return exactly n_features
    if len(all_features) > n_features:
        return all_features[:n_features]
    else:
        return np.pad(all_features, (0, n_features - len(all_features)))

def load_and_process_data(file_path: str, n_features: int = 258) -> Tuple[np.ndarray, List[str]]:
    data = pd.read_csv(file_path, header=None, names=['Hex'])
    logging.info(f"Data shape: {data.shape}")
    
    X_features = []
    skipped_values = []
    
    for hex_value in data['Hex']:
        hex_string = process_hex_data(hex_value.strip())
        try:
            byte_data = hex_to_bytes(hex_string)
            features = extract_features(byte_data, n_features)
            X_features.append(features)
        except ValueError as e:
            skipped_values.append(f"{hex_string}: {str(e)}")
    
    return np.array(X_features), skipped_values

def create_deep_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def combined_predict(features, traditional_model, deep_model, scaler, pca):
    expected_features = scaler.n_features_in_
    if features.shape[1] != expected_features:
        logging.warning(f"Feature mismatch: got {features.shape[1]}, expected {expected_features}")
        if features.shape[1] > expected_features:
            features = features[:, :expected_features]
        else:
            features = np.pad(features, ((0, 0), (0, expected_features - features.shape[1])))
    
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    trad_probs = traditional_model.predict_proba(features_pca)
    deep_probs = deep_model.predict(features_scaled)
    
    return (trad_probs + deep_probs) / 2

def analyze_cipher5(file_path: str, combined_predict, traditional_model, deep_model, scaler, pca, label_encoder, n_features):
    features = process_cipher5_data(file_path, n_features)
    
    combined_probabilities = combined_predict(features, traditional_model, deep_model, scaler, pca)
    combined_prediction = np.argmax(combined_probabilities, axis=1)
    
    predicted_label = label_encoder.inverse_transform(combined_prediction)[0]
    
    top_3_idx = np.argsort(combined_probabilities[0])[-3:][::-1]
    top_3_labels = label_encoder.inverse_transform(top_3_idx)
    top_3_probs = combined_probabilities[0][top_3_idx]
    
    print(f"\nPrediction for cipher5.csv:")
    print(f"Predicted Algorithm: {predicted_label}")
    print("Top 3 Predictions:")
    for label, prob in zip(top_3_labels, top_3_probs):
        print(f"Algorithm {label}: {prob:.4f}")
    
    # Feature importance analysis
    feature_importance = abs(pca.components_).sum(axis=0)
    feature_importance = feature_importance / feature_importance.sum()
    sorted_idx = feature_importance.argsort()[::-1]
    
    feature_names = [
        "Entropy", "Mean", "Std", "Length", "Median", "Skew", "Kurtosis", "Max", "Min", "Range"
    ] + [f"Byte_Freq_{i}" for i in range(256)] + [
        f"Autocorr_{lag}" for lag in [1, 2, 4, 8]
    ] + ["Kolmogorov_Complexity"] + [
        f"{n}-gram_Entropy" for n in [2, 3, 4]
    ] + [f"FFT_{i}" for i in range(5)] + [
        "Diff_Mean", "Diff_Std", "Diff_Skew", "Diff_Kurtosis"
    ]
    
    print("\nTop 10 most important features:")
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[feature_names[i] for i in sorted_idx[:20]], y=feature_importance[sorted_idx[:20]])
    plt.title("Top 20 Most Important Features for Cipher5")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Visualize prediction probabilities
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(range(len(combined_probabilities[0]))), y=combined_probabilities[0])
    plt.title("Prediction Probabilities for Cipher5")
    plt.xlabel("Algorithm Index")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from scipy.stats import entropy, skew, kurtosis
from collections import Counter
import zlib
from typing import List, Tuple
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging

# ... (keep all the utility functions from the previous version) ...

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_mlp_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_deep_model(model, X_train, y_train, X_val, y_val, model_name):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model

def combined_predict(features, traditional_model, deep_models, scaler, pca):
    expected_features = scaler.n_features_in_
    if features.shape[1] != expected_features:
        logging.warning(f"Feature mismatch: got {features.shape[1]}, expected {expected_features}")
        if features.shape[1] > expected_features:
            features = features[:, :expected_features]
        else:
            features = np.pad(features, ((0, 0), (0, expected_features - features.shape[1])))
    
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    
    trad_probs = traditional_model.predict_proba(features_pca)
    deep_probs = [model.predict(features_scaled) for model in deep_models]
    
    # Combine probabilities (you can adjust weights if needed)
    combined_probs = np.mean([trad_probs] + deep_probs, axis=0)
    
    return combined_probs

def process_cipher5_data(file_path: str, n_features: int = 258) -> np.ndarray:
    df = pd.read_csv(file_path)
    logging.info("Columns in the CSV file:")
    logging.info(df.columns)
    
    column_name = df.columns[0]
    logging.info(f"Using column '{column_name}' for hex data.")
    
    hex_data = ''.join(df[column_name].astype(str))
    processed_hex_data = process_hex_data(hex_data)
    logging.info(f"Combined hex data (first 50 characters): {processed_hex_data[:50]}...")
    
    features = extract_features(bytes.fromhex(processed_hex_data), n_features)
    logging.info(f"Number of features generated: {len(features)}")
    
    return features.reshape(1, -1)

if __name__ == "__main__":
    try:
        n_features = 258  # Set this to the desired number of features
        
        # Load and process training data
        X_features, skipped_values = load_and_process_data('hex_data.csv', n_features)
        logging.info(f"Number of features in training data: {X_features.shape[1]}")
        logging.info(f"Processed {len(X_features)} samples")
        logging.info(f"Skipped {len(skipped_values)} invalid samples")
        
        # Handle NaN values and scale features
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Apply dimensionality reduction
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        logging.info(f"PCA reduced feature space: {X_pca.shape[1]} dimensions")
        
        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_pca)
        
        gmm = GaussianMixture(n_components=5, random_state=42)
        gmm_labels = gmm.fit_predict(X_pca)
        
        # Train the traditional model (using GMM for this example)
        traditional_model = gmm
        labels = gmm_labels
        
        # Create and fit label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        # Split the data for deep learning models
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        
        # Create and train multiple deep learning models
        cnn_model = create_cnn_model(X_scaled.shape[1:], num_classes)
        lstm_model = create_lstm_model(X_scaled.shape[1:], num_classes)
        mlp_model = create_mlp_model(X_scaled.shape[1:], num_classes)
        
        deep_models = [
            train_deep_model(cnn_model, X_train, y_train, X_val, y_val, "CNN"),
            train_deep_model(lstm_model, X_train, y_train, X_val, y_val, "LSTM"),
            train_deep_model(mlp_model, X_train, y_train, X_val, y_val, "MLP")
        ]
        
        # Evaluate the ensemble model
        ensemble_predictions = combined_predict(X_val, traditional_model, deep_models, scaler, pca)
        ensemble_accuracy = accuracy_score(y_val, np.argmax(ensemble_predictions, axis=1))
        logging.info(f'Ensemble model validation accuracy: {ensemble_accuracy:.4f}')
        
        # Create confusion matrix
        cm = confusion_matrix(y_val, np.argmax(ensemble_predictions, axis=1))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Ensemble Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Save all models and preprocessing objects
        joblib.dump(traditional_model, 'crypto_identifier_traditional_model.joblib')
        joblib.dump(scaler, 'crypto_identifier_scaler.joblib')
        joblib.dump(pca, 'crypto_identifier_pca.joblib')
        joblib.dump(label_encoder, 'crypto_identifier_label_encoder.joblib')
        for i, model in enumerate(deep_models):
            model.save(f'crypto_identifier_deep_model_{i}.h5')
        logging.info("All models and preprocessing objects saved.")

        # Analyze cipher5 using the combined model
        try:
            analyze_cipher5('cipher5.csv', combined_predict, traditional_model, deep_models, scaler, pca, label_encoder, n_features)
        except Exception as e:
            logging.error(f"Error analyzing cipher5: {str(e)}")
            logging.error("Traceback:", exc_info=True)

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        logging.error("Traceback:", exc_info=True)


# In[ ]:




