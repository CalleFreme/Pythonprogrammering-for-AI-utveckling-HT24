# Vilken clustering-metod ska jag använda?

# K-means
# När klustren förväntas vara sfäriska
# När datastorleken är stor (skalbar)
# När vi vet ungefärligt antal kluster
# När vi vill ha enkla, tolkningsbara resultat

# DBSCAN
# När klustren är oregelbundna
# När outliers/noise i data
# När vi inte vet antal kluster i förväg
# När klustren har olika densitet

# Hierarchical clustering
# Vi vill förstå hierarkin i datan
# När datastorleken är måttlig (<10000 punkter)
# När vi vill ha flexibilitet i antal kluster
# När behöver detaljerad insikt i klusterstrukturen

# Gaussian Mixture Models
# När klustren kan överlappa
# När vi vill ha sannolikhetsbaserade tillhörigheter
# När data följer normalfördelning
# När vi behöver "soft" clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from typing import List, Tuple, Optional
import logging

# Konfigurerar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        '''
        Förbereder data genom att hantera saknade värden och standardisera
        '''
        # Saknde värden
        data_cleaned = data.fillna(data.mean())

        # Standardisera data
        return self.scaler.fit_transform(data_cleaned)

    def reduce_dimensions(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        '''
        Reducerar dimensinoer med PCA
        '''
        self.pca = PCA(n_components=n_components)
        reduced_data = self.pca.fit_transform(data)
        logger.info(f"Förklarad varians: {self.pca.explained_variance_ratio_}")
        return reduced_data


class KMeansAnalyzer:
    '''
    Klass för att utföra K-means clustering och analysera resultat
    
    K-means antar att att kluster är sfäriska och har ungefär samma densitet (kompakthet) och varians.
    Särskilt bra om vi vet det förvaften antalet kluster.
    Försöker minimera summan av kvadrerade avstånd (sum of squared distances) inom varje kluster.
    Hanterar komplexa och oregelbundna klusterformer mindre bra.
    Hanterar outliers mindre bra, leder till oregelbundna klusterformer och sãoligt positionerade centroider.

    Eftersom K-means kräver att vi vet antalet kluster vi vill dela upp datapunkterna i, försöker vi först hitta ett optimalt antal kluster.
    '''

    def __init__(self, max_clusters: int = 10):
        self.max_cluster = max_clusters
        self.kmeans_dict = {}
        self.best_kmeans = None
        self.optimal_clusters = None

    def find_optimal_clusters(self, data: np.ndarray) -> int:
        '''
        Hittar optimalt antal kluster med elbow-metoden.
        Går iterativt igenom stigande värden på k, för att hitta det värde som ger bäst resultat.

        För varje k, skapar vi en ny K-means-modell, tränar den och sparar i self.kmeans_dict för att kunna använda den senare.
        "intertia", "sum of squared distances of samples to their closest cluster center", beräknas för att avgöra vår "elbow" point,
        som representerar det optimala antalet kluster.
        '''

        inertias = []

        for k in range(1, self.max_cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            self.kmeans_dict[k] = kmeans
            inertias.append(kmeans.inertia_)

        # Visualisera elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_cluster + 1), inertias, 'bx-')
        plt.xlabel('k (antal kluster)')
        plt.ylabel('Inertia')
        plt.title('Elbow-metoden för optimalt antal kluster')
        plt.show()

        # Automatisk detekteriong av elbow point
        diffs = np.diff(inertias)
        elbow_point = np.argmin(diffs) + 1

        self.optimal_clusters = elbow_point
        self.best_kmeans = self.kmeans_dict[elbow_point]    # Sparar den bästa k-means-modellen i self.best_kmeans

        return elbow_point

    def cluster_data(self, data: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        '''
        Utför klustring med angivet eller optimalt antal kluster

        Params:
            data: Input data
            n_clusters: Antal kluster (optional)

        Returns:
            Cluster labels
        '''
        if n_clusters is None:
            if self.optimal_clusters is None:
                self.find_optimal_clusters(data)
            n_clusters = self.optimal_clusters  # Plockar fram optimala antalet kluster som beräknats tidigare

        kmeans = KMeans(n_clusters=n_clusters, random_state=42) # 42 (kan vara annat värde) eftersom vi vill ha slump men "samma" resultat varje körning
        return kmeans.fit_predict(data)

    def visualize_clusters(self, data: np.ndarray, labels: np.ndarray):
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.xlabel('Första komponenten')
        plt.ylabel('Andra komponenten')
        plt.title('K-means klustringsresultat')
        plt.show()

class DBSCANAnalyzer:
    '''
    Klass för att utföra DBSCAN för oregelbundna former (Density-Based Spatial Clustering of Applications with Noise).

    Bra för kuster som inte nödvändigtvis är sfäriskt formade.
    Hittar kluster baserat på densitet, identifierar punkter som har ett minimalt antal grann-punkter (neighboring points)
    inom ett specificerat avstånd (epsilon).
    DBSCAN behöver inte veta antalet kluster, men kräver paremeters eps (epsilon) och min_samples (minsta antalet punkter i ett "neighborhood" för att skapa en "dense region").

    DBSCAN kan identifiera "noise points", punkter som inte tillhör något kluster. Hanterarält outlier bättre föran K-means.

    '''

    def __init__(self):
        self.dbscan = None

    def find_optimal_params(self, data: np.ndarray) -> Tuple[float, int]:
        '''
        Hittar optimala parametrar för DBSCAN

        Beräknar epsilon-värdet som krävs för DBSCAN. Epsilon motsvarar "neighborhood radius", d.v.s det radie inom vilken punkter ska räknas som "grannar".
        För att en datapunkt ska vara del av ett kluster, måste tillräckligt många andra punkter (definierat av 'min_samples') vara inom radien 'eps'.
        Den andra parametern min_samples bestäms ofta till det dubbla av antalet features för att ge tillräckligt hög täthet i kluster.

        Om värdet på eps är för litet, kommer färre punkter räknas som grannar, ger för mycket noise och fragmenterade kluster.
        Om bärdet på eps är för stort, räknas för många punkter som grannar och vi får färre distinkta kluster.

        Använder först NearestNeighbor för att hitta de närmsta grann-avstånden för varje datapunkt.
        Avstånden sorteras i stigande (ascending) ordning (och visualiserar avstånden till varje punkt närmaste granne)
        Den "elbow" point som kan identifieras i denna graf, där avstånden börjar öka i snabb takt, antyder om ett lämpligt värde för epsilon.
        Vid elbow point hittar vi alltså tröskeln mellan täta regioner (kluster) och glesa områden (noise)

        Punkter delas in i "core points", "border points" och "noise/outlier points".
        En core point har minst min_samples antal punkter (inkl. sig själv) inom radien eps.
        En border point har mindre än min_samples antal punkter inom radien eps., men är granne till en core point.
        En noise/outlier point är varken core eller border point.
        '''
        # Beräkna närmaste grannar för varje punkt
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        # Sortera avstånden för att hitta "elbow"
        distances = np.sort(distances[:, 1])
        
        # Visualisera för att hitta eps
        # En s.k. k-distance graph som visar elbow point
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel('Punkter')
        plt.ylabel('Avstånd till närmaste granne')
        plt.title('K-distance Graf för DBSCAN Parameter Selection')
        plt.show()
        
        # Föreslå eps baserat på "elbow point"
        knee_point = np.diff(distances).argmax()
        eps = distances[knee_point]
        min_samples = 2 * data.shape[1]  # Tumregel: 2 * antal dimensioner
        
        return eps, min_samples

    def cluster_data(self, data: np.ndarray, eps: Optional[float] = None, min_samples: Optional[int] = None) -> np.ndarray:
        '''
        utför DBSCAN klustring

        Params:
            data: input data
            eps: epsilon parameter
            min_samples: Minimum samples parameter
        Returns:
            Cluster labels
        '''
        if eps is None or min_samples is None:
            eps, min_samples = self.find_optimal_params(data)

        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return self.dbscan.fit_predict(data)

def main():
    '''
    Huvudfunktion
    '''

    # Skapar syntetisk data för demo
    np.random.seed(42)

    # Generera tre kluster med olika former
    n_samples = 300

    # Första kluster: cirkulärt
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples//3, 2))

    # Andra klustret: elliptiskt
    cluster2 = np.random.normal(loc=[-2, -2], scale=[1, 0.3], size=(n_samples//3, 2))

    # Tredje klustret: linjärt
    x = np.linspace(-1, 1, n_samples//3)
    y = 2*x + np.random.normal(0, 0.1, n_samples//3)
    cluster3 = np.column_stack((x, y))

    # Kombinera kluster
    X = np.vstack([cluster1, cluster2, cluster3])

    # Initiera preproccesor
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.prepare_data(pd.DataFrame(X))

    # Demonstrera K-means
    logger.info("Utför K-means clustering...")
    kmeans_analyzer = KMeansAnalyzer(max_clusters=10)
    optimal_k = kmeans_analyzer.find_optimal_clusters(X_scaled)
    kmeans_labels = kmeans_analyzer.cluster_data(X_scaled)
    kmeans_analyzer.visualize_clusters(X_scaled, kmeans_labels)

    # Demonstrera DBSCAN
    logger.info("Utför DBSCAN clustering...")
    dbscan_analyzer = DBSCANAnalyzer()
    dbscan_labels = dbscan_analyzer.cluster_data(X_scaled)

    # Visualisering av DBSCAN resultat
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('Första komponenten')
    plt.ylabel('Andra komponenten')
    plt.title('DBSCAN klustringsresultat')
    plt.show()

    # Jämför reseltatet
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
    ax1.set_title('K-means Resultat')
    
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
    ax2.set_title('DBSCAN Resultat')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
