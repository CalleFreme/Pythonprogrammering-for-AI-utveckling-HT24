# Övning 6: Klustring med Scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generera syntetisk data
n_samples = 300
n_features = 2
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Implementera K-means klustring
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Visualisera resultaten
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering Results')
plt.colorbar(scatter)
plt.show()

# Experimentera med olika antal kluster
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 7))
plt.plot(k_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Kommentarer:
# 1. Vi använder make_blobs för att generera syntetisk data med tydliga kluster.
# 2. KMeans-algoritmen grupperar datapunkter i k kluster baserat på deras likhet.
# 3. Visualiseringen visar hur datapunkterna har grupperats, med olika färger för varje kluster.
# 4. Klustercentrumen markeras med röda X för att visa var algoritmen har placerat klustermittpunkterna.
# 5. Vi experimenterar med olika antal kluster (k) och plottar inertia (summan av kvadrerade avstånd till närmaste centrumpunkt) för varje k.
# 6. Armbågsmetoden (elbow method) används för att hitta det optimala antalet kluster, där kurvan böjer sig som en armbåge.