import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
import joblib
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
from datetime import datetime
import h5py

# Konfigurera logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Klass för att hantera sparning och laddning av olika modelltyper.
    """
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def save_model(self, model: any, model_name: str, model_type: str):
        """
        Sparar en modell i lämpligt format baserat på typ.
        
        Args:
            model: Modellen som ska sparas
            model_name: Namnet på modellen
            model_type: Typ av modell ('sklearn', 'numpy', 'custom')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(self.base_path, f"{model_name}_{timestamp}")
        
        try:
            if model_type == 'sklearn': # Scikit-learn används joblib-filformat
                joblib.dump(model, f"{full_path}.joblib")
            elif model_type == 'numpy': # för att spara NumPy-arrays
                np.save(f"{full_path}.npy", model)
            elif model_type == 'custom':    # För generella Python-objekt som sparas i pickle-format
                with open(f"{full_path}.pkl", 'wb') as f:
                    pickle.dump(model, f)
            elif model_type == 'h5':    # HDF5-filer för att spara TensorFlow-modeller
                model.save(f"{full_path}.h5")
            
            # Spara metadata
            metadata = {
                'model_type': model_type,
                'timestamp': timestamp,
                'model_name': model_name
            }
            with open(f"{full_path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Modell sparad: {full_path}")
            
        except Exception as e:
            logger.error(f"Fel vid sparning av modell: {str(e)}")
            raise
    
    def load_model(self, model_path: str, model_type: str):
        """
        Laddar en modell från fil.
        
        Args:
            model_path: Sökväg till modellen
            model_type: Typ av modell
        Returns:
            Laddad modell
        """
        try:
            if model_type == 'sklearn':
                return joblib.load(model_path)
            elif model_type == 'numpy':
                return np.load(model_path)
            elif model_type == 'custom':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            elif model_type == 'h5':
                return load_model(model_path) # Tensorflow case
            
        except Exception as e:
            logger.error(f"Fel vid laddning av modell: {str(e)}")
            raise

class HierarchicalClusterAnalyzer:
    """
    Klass för hierarkisk klustringsanalys med olika visualiseringsmetoder.
    En s.k. "bottom-up", eller agglomerativ klustring, där vi utnyttjar en linkage-metod, default är 'Ward's linkage'. Euklidiskt avstånd.
    """
    def __init__(self):
        self.linkage_matrix = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray, method: str = 'ward', metric: str = 'euclidean'):
        """
        Utför hierarkisk klustring.
        Skalar först input data och applicerar sedan en linkage-metod för att skapa en hierarkisk trä-struktur.
        
        Args:
            data: Input data
            method: Linkage metod ('ward', 'complete', 'average', 'single')
            metric: Distansmått
        """
        scaled_data = self.scaler.fit_transform(data)   # Använder sig av StandardScaler. Resulterar i att: mean (genomsnittsvärde)=0, standard deviation (standardavvikelse)=1 
        self.linkage_matrix = linkage(scaled_data, method=method, metric=metric) # Skapar en linkage matrix, innehåller kluster-information baserat på avståndet mellan datapunkter.
        # Matrisen består alltså av: kluster-par som har slagits ihop/merged, avstånden mellan kluster, och antalet datapunkter i varje kluster
        return self
    
    def get_clusters(self, n_clusters: int) -> np.ndarray:
        """
        Hämtar klusterlabels för specificerat antal kluster.
        Ger varje datapunkt en klusterlabel baserat på den hierarkiska klustringen som sket i fit().
        
        Args:
            n_clusters: Önskat antal kluster
        Returns:
            Array med klusterlabels
        """
        if self.linkage_matrix is None:
            raise ValueError("Kör fit() först!")
            
        self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') # Ger varje datapunkt ett kluster baserat på linkage matrix, antalet kluster vi vill ha. 
        # 'maxclust' gör att vi gör att vi formar exakt n_clusters antal kluster.
        return self.labels  # Returnerar en array av cluster labels, där varje label motsvarar en cluster assignment för en datapunkt.
    
    def plot_dendrogram(self, figure_size: Tuple[int, int] = (10, 7)):
        """
        Plottar dendrogram för att visualisera hierarkin.
        
        Args:
            figure_size: Storlek på plotten
        """
        plt.figure(figsize=figure_size)
        dendrogram(self.linkage_matrix)
        plt.title('Hierarkiskt Klusteringsdendrogram')
        plt.xlabel('Sampel Index')
        plt.ylabel('Avstånd')
        plt.show()
    
    def plot_cluster_heatmap(self, data: np.ndarray):
        """
        Skapar en heatmap av klusterrelationer.
        
        Args:
            data: Original data
        """
        if self.labels is None:
            raise ValueError("Kör get_clusters() först!")
            
        # Sortera data efter klusterlabels
        sorted_indices = np.argsort(self.labels)
        sorted_data = data[sorted_indices]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sorted_data, cmap='viridis')
        plt.title('Kluster Heatmap')
        plt.show()

class RecommenderSystem:
    """
    Enkel rekommendationsmotor baserad på item-item likhet.
    Generarar likhetsbaserade item recommendations baserat på en matris av user-item ratings.
    """
    def __init__(self):
        self.item_similarity_matrix = None
        self.items_df = None
        self.scaler = MinMaxScaler()
        
    def fit(self, ratings_matrix: pd.DataFrame):
        """
        Tränar rekommendationssystemet.
        Normaliserar matrisen av user-item ratings och beräknar item-item similarity.
        
        Args:
            ratings_matrix: DataFrame med user-item ratings
        """
        # Normalisera ratings
        normalized_ratings = self.scaler.fit_transform(ratings_matrix) # Använder sig av MinMaxScaler för att normalisera ratings till värden mellan 0 och 1.
        # Bevarar relativt avstånd mellan datapunkter, bra för träffsäkerheten i similarity-beräkningar.
        
        # Beräkna item-item similarity
        # np.corrcoef beräknar correlations-koefficienten mellan items. 
        self.item_similarity_matrix = np.corrcoef(normalized_ratings.T) # Vi använder '.T' för att transponera matrisen så att vi
        #beräknar korrelation mellan item-item (column-column) istället för user-item (row-column).
        # item_similarity_matrix är alltså en kvadratisk matris där varje cell (i, j) indikerar likheten mellan item i och item j

        self.items_df = ratings_matrix # Sparar ursprungliga ratings-matrisen för att senare kunna komma åt item metadata eller user ratings.
        return self
    
    def get_recommendations(self, item_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Hämtar rekommendationer baserat på ett item.
        Tar fram de items som är mest lika ett givet item, baserat på item-item similarity-matrisen.

        Args:
            item_id: ID för item att basera rekommendationer på
            n_recommendations: Antal rekommendationer att returnera
        Returns:
            Lista med (item_id, similarity_score) tupler
        """
        if self.item_similarity_matrix is None:
            raise ValueError("Kör fit() först!")
            
        # Hämta likheter för item
        item_similarities = self.item_similarity_matrix[item_id]
        
        # Sortera och returnera top-N (exkludera det givna itemet själv)
        # Skapar en lista av tuples (i, sim), där i är ett items index, och sim är dess similarity score
        similar_items = [(i, sim) for i, sim in enumerate(item_similarities) 
                        if i != item_id]
        similar_items.sort(key=lambda x: x[1], reverse=True) # Sorterar i fallande (descending) ordning, baserat på similarity score
        
        return similar_items[:n_recommendations] # Returnerar de n_recommendations (default 5) antal mest liknande items
    
    def plot_similarity_heatmap(self):
        """
        Visualiserar item-item similarity matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.item_similarity_matrix, cmap='coolwarm', center=0)
        plt.title('Item-Item Similarity Matrix')
        plt.show()

def main():
    """
    Huvudfunktion som demonstrerar användning av klasserna.
    """
    # Skapa exempel-data för hierarkisk klustring
    np.random.seed(42)
    n_samples = 150
    
    # Generera tre distinkta grupper av data
    group1 = np.random.normal(loc=[0, 0], scale=0.3, size=(n_samples//3, 2))
    group2 = np.random.normal(loc=[2, 2], scale=0.3, size=(n_samples//3, 2))
    group3 = np.random.normal(loc=[-1, 2], scale=0.3, size=(n_samples//3, 2))
    
    X = np.vstack([group1, group2, group3])
    
    # Demonstrera hierarkisk klustring
    logger.info("Utför hierarkisk klustring...")
    hc_analyzer = HierarchicalClusterAnalyzer()
    hc_analyzer.fit(X)
    hc_analyzer.plot_dendrogram()
    
    # Få kluster och visualisera
    labels = hc_analyzer.get_clusters(n_clusters=3)
    hc_analyzer.plot_cluster_heatmap(X)
    
    # Skapa exempel-data för rekommendationssystem
    n_users = 100
    n_items = 50
    sparsity = 0.2  # Hur många ratings som saknas
    
    # Generera synthetic ratings matrix
    ratings = np.random.normal(loc=3.5, scale=1.0, size=(n_users, n_items))
    ratings = np.clip(ratings, 1, 5)  # Begränsa ratings till 1-5
    
    # Lägg till lite sparsity
    mask = np.random.random(ratings.shape) < sparsity
    ratings[mask] = 0
    
    # Konvertera till DataFrame
    ratings_df = pd.DataFrame(ratings)
    
    # Demonstrera rekommendationssystem
    logger.info("Tränar rekommendationssystem...")
    recommender = RecommenderSystem()
    recommender.fit(ratings_df)
    
    # Visa några rekommendationer
    test_item = 0
    recommendations = recommender.get_recommendations(test_item)
    logger.info(f"Top 5 rekommendationer för item {test_item}:")
    for item_id, similarity in recommendations:
        logger.info(f"Item {item_id}: Similarity = {similarity:.3f}")
    
    # Visualisera similarity matrix
    recommender.plot_similarity_heatmap()
    
    # Demonstrera model saving/loading
    model_manager = ModelManager()
    
    # Spara modeller
    logger.info("Sparar modeller...")
    model_manager.save_model(hc_analyzer, "hierarchical_clustering", "custom")
    model_manager.save_model(recommender, "recommender_system", "custom")
    
    # Exempel på laddning
    logger.info("Testar modell-laddning...")
    loaded_model = model_manager.load_model(
        "models/hierarchical_clustering_latest.pkl", 
        "custom"
    )

if __name__ == "__main__":
    main()