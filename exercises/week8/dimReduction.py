# Även om PCA är användbart, föredras ofta t-distributed Stochasti Neighbor Embedding (t-SNE) för att
# visualisera komplex data i lägre dimensioner. Detta eftersom metoden fångar icke-linjära relationer och
# bevarar kluster-strukturen.
# Mål: Använd t-SNE för att reducerar dimensionaliteten av hög-dimensionell data, så som MNIST datasetet (handskriva siffror),
# för effektivare visualisering.
# Metod: Använd TSNE fårn sklearn.manifold

# 1. Ladda in high-dimensioal data
# 2. Applicera t-SNE för att reducera datat till 2D.
# 3. Plotta resultatet för att visuellt identifiera mönster och kluster.

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Ladda data
digits = load_digits()
X = digits.data # 64-dimensionellt dataset
y = digits.target

# Applicera t-SNE för att reducera dimensionen till 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_reduced = tsne.fit_transform(X)

# Visualisera 2D-projektionen med färg-kodade kluster
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Spectral', alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title("t-SNE Visualization of the MNIST Digits Dataset")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
