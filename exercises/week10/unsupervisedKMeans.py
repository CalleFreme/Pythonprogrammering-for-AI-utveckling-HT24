import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Build k-means clustering model
k = 3  # Number of clusters
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=k, use_mini_batch=False)

# Input function
input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": data}, num_epochs=1, shuffle=False)

# Train model
kmeans.train(input_fn)

# Predict cluster assignments
cluster_assignments = list(kmeans.predict_cluster_index(input_fn))
print("Cluster assignments:", cluster_assignments)
