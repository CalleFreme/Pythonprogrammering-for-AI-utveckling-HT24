from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Generera syntetisk data med överlappand kluster
np.random.seed(42)
X1 = np.random.normal(loc=[5, 5], scale=1.5, size=(100, 2))
X2 = np.random.normal(loc=[10, 10], scale=1.5, size=(100, 2))
X3 = np.random.normal(loc=[5, 10], scale=1.5, size=(100, 2))
X = np.vstack([X1, X2, X3])

# Applicera GMM med 3 components
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)  # Get cluster membership probabilities

# Plotta kluster med sannolikheter indikerade med färg-intensitet
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Gaussian Mixture Model Clustering with Soft Labels")
plt.show()
