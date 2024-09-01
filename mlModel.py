#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().run_line_magic('pip', 'install umap-learn')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from scipy.stats import entropy
from collections import Counter
import zlib
from sklearn.manifold import TSNE
from umap import UMAP


def hex_to_bytes(hex_string):
    return bytes.fromhex(hex_string)

def bytes_to_int_array(byte_data):
    return np.frombuffer(byte_data, dtype=np.uint8)

def calculate_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return entropy(probabilities)

def calculate_byte_frequency(data):
    return np.bincount(data, minlength=256) / len(data)

def autocorrelation(data, lag=1):
    return np.corrcoef(data[:-lag], data[lag:])[0, 1]

def kolmogorov_complexity(data):
    return len(zlib.compress(data)) / len(data)

def n_gram_analysis(data, n):
    ngrams = [''.join(map(str, data[i:i+n])) for i in range(len(data)-n+1)]
    return entropy(list(Counter(ngrams).values()))

def extract_features(byte_data):
    int_data = bytes_to_int_array(byte_data)
    
    # Basic statistical features
    basic_features = [
        calculate_entropy(int_data),
        np.mean(int_data),
        np.std(int_data),
        len(int_data)
    ]
    
    # Byte frequency
    byte_freq = calculate_byte_frequency(int_data)
    
    # Autocorrelation
    autocorr = [autocorrelation(int_data, lag) for lag in [1, 2, 4]]
    
    # Kolmogorov complexity estimation
    kol_complexity = kolmogorov_complexity(byte_data)
    
    # N-gram analysis
    ngram_features = [n_gram_analysis(int_data, n) for n in [2, 3, 4]]
    
    # Combine all features
    return np.array(basic_features + list(byte_freq) + autocorr + [kol_complexity] + ngram_features)

# Load and process the data
data = pd.read_csv('hex_data.csv', header=None, names=['Hex'])
print("Data shape:", data.shape)

X_features = []
for hex_value in data['Hex']:
    hex_string = hex_value.strip()
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string
    
    try:
        byte_data = hex_to_bytes(hex_string)
        features = extract_features(byte_data)
        X_features.append(features)
    except ValueError as e:
        print(f"Skipping invalid hex value: {hex_string}. Error: {e}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# Convert list of features to numpy array
X_features = np.array(X_features)
print(f"Feature array shape: {X_features.shape}")

# Check for NaN values
nan_counts = np.isnan(X_features).sum(axis=0)
print("NaN counts per feature:")
print(nan_counts)

# Impute NaN values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_features)

# Normalize the features using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply dimensionality reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced feature space: {X_pca.shape[1]} dimensions")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
print("t-SNE applied")

# Apply UMAP
umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(X_scaled)
print("UMAP applied")

# Clustering algorithms
# KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca)

# DBSCAN
# Note: You might need to adjust eps and min_samples based on your data
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(X_pca)

# Print clustering results
print("\nClustering Results:")
print(f"K-Means: {len(np.unique(kmeans_labels))} clusters")
print(f"DBSCAN: {len(np.unique(dbscan_labels))} clusters (including noise)")
print(f"GMM: {len(np.unique(gmm_labels))} clusters")

# Optional: Add silhouette score calculation
from sklearn.metrics import silhouette_score

for name, labels in [("K-Means", kmeans_labels), ("DBSCAN", dbscan_labels), ("GMM", gmm_labels)]:
    if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 labels
        score = silhouette_score(X_pca, labels)
        print(f"{name} Silhouette Score: {score:.4f}")

# Evaluate clustering
def evaluate_clustering(X, labels, algorithm_name):
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"{algorithm_name} Silhouette Score: {score:.4f}")
    else:
        print(f"{algorithm_name}: All samples assigned to one cluster")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(f'{algorithm_name} Clustering')
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.colorbar(scatter, label='Cluster')
    if algorithm_name == 'DBSCAN':
        noise_mask = labels == -1
        plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', marker='x', s=100, label='Noise')
        plt.legend()
    plt.show()

print("\nK-Means Evaluation:")
evaluate_clustering(X_pca, kmeans_labels, "K-Means")
evaluate_clustering(X_tsne, kmeans_labels, "K-Means (t-SNE)")
evaluate_clustering(X_umap, kmeans_labels, "K-Means (UMAP)")

print("\nDBSCAN Evaluation:")
evaluate_clustering(X_pca, dbscan_labels, "DBSCAN")
evaluate_clustering(X_tsne, dbscan_labels, "DBSCAN (t-SNE)")
evaluate_clustering(X_umap, dbscan_labels, "DBSCAN (UMAP)")

print("\nGaussian Mixture Model Evaluation:")
evaluate_clustering(X_pca, gmm_labels, "GMM")
evaluate_clustering(X_tsne, gmm_labels, "GMM (t-SNE)")
evaluate_clustering(X_umap, gmm_labels, "GMM (UMAP)")

# Analyze cluster characteristics
def analyze_clusters(X_original, labels, algorithm_name):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1 or (algorithm_name == 'DBSCAN' and label == -1):
            cluster_data = X_original[labels == label]
            print(f"\n{algorithm_name} - {'Noise' if label == -1 else f'Cluster {label}'} characteristics:")
            print(f"Number of samples: {len(cluster_data)}")
            print(f"Average entropy: {np.mean(cluster_data[:, 0]):.4f}")
            print(f"Average byte value: {np.mean(cluster_data[:, 1]):.4f}")
            print(f"Average standard deviation: {np.mean(cluster_data[:, 2]):.4f}")
            print(f"Average length: {np.mean(cluster_data[:, 3]):.4f}")
            
            avg_byte_freq = np.mean(cluster_data[:, 4:260], axis=0)
            plt.figure(figsize=(12, 6))
            plt.bar(range(256), avg_byte_freq)
            plt.title(f"{algorithm_name} - {'Noise' if label == -1 else f'Cluster {label}'} Average Byte Frequency Distribution")
            plt.xlabel("Byte Value")
            plt.ylabel("Frequency")
            plt.show()

analyze_clusters(X_features, kmeans_labels, "K-Means")
analyze_clusters(X_features, dbscan_labels, "DBSCAN")
analyze_clusters(X_features, gmm_labels, "GMM")

# Print cluster sizes
print("\nCluster sizes:")
print("K-Means:", np.bincount(kmeans_labels))

# Handle DBSCAN labels (which may include -1 for noise)
dbscan_sizes = np.bincount(dbscan_labels[dbscan_labels >= 0])
print("DBSCAN:")
print(" Clusters:", dbscan_sizes)
print(" Noise points:", np.sum(dbscan_labels == -1))

print("GMM:", np.bincount(gmm_labels))

# Additional analysis for DBSCAN
unique_labels = np.unique(dbscan_labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
print(f"\nDBSCAN formed {n_clusters} clusters")
print(f"Percentage of points classified as noise: {np.mean(dbscan_labels == -1) * 100:.2f}%")

# Visualize DBSCAN results with noise points highlighted
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering with Noise')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(scatter, label='Cluster')
noise_mask = dbscan_labels == -1
plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], c='red', marker='x', s=100, label='Noise')
plt.legend()
plt.show()


# Additional analysis: Print feature importance
feature_importance = abs(pca.components_).sum(axis=0)
feature_importance = feature_importance / feature_importance.sum()
sorted_idx = feature_importance.argsort()[::-1]
print("\nTop 10 most important features:")
for idx in sorted_idx[:10]:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib

# Assuming X_features, kmeans_labels are already defined

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or 'most_frequent' as well
X_imputed = imputer.fit_transform(X_features)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(kmeans_labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Train ensemble model
ensemble = VotingClassifier(
    estimators=[('rf', RandomForestClassifier(random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
                ('nn', MLPClassifier(random_state=42, max_iter=1000))],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Evaluate on test data
y_pred = ensemble.predict(X_test)
print("Ensemble performance on test data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Cross-validation
cv_scores = cross_val_score(ensemble, X_pca, y_encoded, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    ensemble, X_pca, y_encoded, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.show()

# Feature Importance (using Random Forest)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_pca, y_encoded)

importances = permutation_importance(rf_classifier, X_pca, y_encoded, n_repeats=10, random_state=42)
feature_importance = importances.importances_mean
feature_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.title('Feature Importance')
plt.xlabel('Principal Components')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Save the model and preprocessing objects
joblib.dump(ensemble, 'crypto_identifier_model.joblib')
joblib.dump(scaler, 'crypto_identifier_scaler.joblib')
joblib.dump(pca, 'crypto_identifier_pca.joblib')
joblib.dump(label_encoder, 'crypto_identifier_label_encoder.joblib')

# Simple Prediction Interface
def predict_algorithm(hex_data):
    # Load saved objects
    loaded_model = joblib.load('crypto_identifier_model.joblib')
    loaded_scaler = joblib.load('crypto_identifier_scaler.joblib')
    loaded_pca = joblib.load('crypto_identifier_pca.joblib')
    loaded_label_encoder = joblib.load('crypto_identifier_label_encoder.joblib')

    # Process hex data
    features = extract_features(bytes.fromhex(hex_data))
    
    # Impute missing values if any
    features_imputed = imputer.transform(features.reshape(1, -1))
    
    features_scaled = loaded_scaler.transform(features_imputed)
    features_pca = loaded_pca.transform(features_scaled)

    # Predict
    prediction = loaded_model.predict(features_pca)
    probabilities = loaded_model.predict_proba(features_pca)

    # Decode prediction
    predicted_label = loaded_label_encoder.inverse_transform(prediction)[0]

    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
    top_3_labels = loaded_label_encoder.inverse_transform(top_3_idx)
    top_3_probs = probabilities[0][top_3_idx]

    return predicted_label, list(zip(top_3_labels, top_3_probs))

# Test the prediction interface
test_hex = data['Hex'].iloc[0]  # Using the first sample from your dataset
predicted_label, top_3_predictions = predict_algorithm(test_hex)

print("\nPrediction Interface Test:")
print(f"Predicted Cluster: {predicted_label}")
print("Top 3 Predictions:")
for label, prob in top_3_predictions:
    print(f"Cluster {label}: {prob:.4f}")


# In[9]:


import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Helper functions
def hex_to_bytes(hex_string):
    return bytes.fromhex(hex_string)

def bytes_to_int_array(byte_data):
    return np.frombuffer(byte_data, dtype=np.uint8)

def calculate_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def calculate_byte_frequency(data):
    return np.bincount(data, minlength=256) / len(data)

def process_hex_data(hex_string):
    hex_string = ''.join(hex_string.split())
    hex_string = ''.join(c for c in hex_string if c in '0123456789ABCDEFabcdef')
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string
    return hex_string

def extract_features(byte_data):
    int_data = bytes_to_int_array(byte_data)
    byte_freq = calculate_byte_frequency(int_data)
    return np.array([
        calculate_entropy(int_data),
        np.mean(int_data),
        np.std(int_data),
        len(int_data),
        *byte_freq
    ])

def predict_algorithm(hex_data):
    # Load saved objects
    loaded_model = joblib.load('crypto_identifier_model.joblib')
    loaded_scaler = joblib.load('crypto_identifier_scaler.joblib')
    loaded_pca = joblib.load('crypto_identifier_pca.joblib')
    loaded_label_encoder = joblib.load('crypto_identifier_label_encoder.joblib')
    
    # Process hex data
    processed_hex = process_hex_data(hex_data)
    features = extract_features(bytes.fromhex(processed_hex))
    
    # Check if the number of features matches the expected number
    expected_features = loaded_scaler.n_features_in_
    if len(features) != expected_features:
        print(f"Warning: Number of extracted features ({len(features)}) "
              f"does not match the expected number ({expected_features}).")
        
        if len(features) < expected_features:
            # Pad with zeros if we have fewer features than expected
            features = np.pad(features, (0, expected_features - len(features)))
        else:
            # Truncate if we have more features than expected
            features = features[:expected_features]
        
        print(f"Adjusted number of features to {len(features)}.")
    
    features_scaled = loaded_scaler.transform(features.reshape(1, -1))
    features_pca = loaded_pca.transform(features_scaled)
    
    # Predict
    prediction = loaded_model.predict(features_pca)
    probabilities = loaded_model.predict_proba(features_pca)
    
    # Decode prediction
    predicted_label = loaded_label_encoder.inverse_transform(prediction)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
    top_3_labels = loaded_label_encoder.inverse_transform(top_3_idx)
    top_3_probs = probabilities[0][top_3_idx]
    
    return predicted_label, list(zip(top_3_labels, top_3_probs))

# Main execution
if __name__ == "__main__":
    # Read data from cipher5.csv
    df = pd.read_csv('cipher5.csv')
    print("Columns in the CSV file:")
    print(df.columns)
    
    # Get the name of the first column
    column_name = df.columns[0]
    print(f"Using column '{column_name}' for hex data.")
    
    # Combine all values in the column
    hex_data = ''.join(df[column_name].astype(str))
    
    # Process the hex data
    processed_hex_data = process_hex_data(hex_data)
    print(f"Combined hex data (first 50 characters): {processed_hex_data[:50]}...")
    
    # Check the number of features generated
    features = extract_features(bytes.fromhex(processed_hex_data))
    print(f"Number of features generated: {len(features)}")
    
    # Make prediction for the combined hex data
    predicted_label, top_3_predictions = predict_algorithm(processed_hex_data)
    
    print(f"\nPrediction for the entire '{column_name}' column:")
    print(f"Predicted Cluster: {predicted_label}")
    print("Top 3 Predictions:")
    for label, prob in top_3_predictions:
        print(f"Cluster {label}: {prob:.4f}")
    
    print("\nPrediction completed for cipher5.csv")


# In[ ]:




