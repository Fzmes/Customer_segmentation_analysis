import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Définir le chemin vers le fichier CSV et charger les données
file_path = os.path.join("E-commerce Customer behavior.csv")
data = pd.read_csv(file_path)
print("Données importées avec succès.")

# Afficher les premières lignes des données
print(data.head())

# Sélectionner les colonnes pertinentes et supprimer les lignes manquantes
data = data[['Total Spend', 'Items Purchased', 'Average Rating', 'Satisfaction Level']].dropna()
print(f"Données nettoyées : {data.shape[0]} lignes restantes.")

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Total Spend', 'Items Purchased', 'Average Rating']])
print("Données normalisées.")

# Création du dendrogramme pour visualiser les clusters potentiels
plt.figure(figsize=(10, 7))
plt.title("Dendrogramme pour le Clustering Hiérarchique")
dendrogram = sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.xlabel("Index des clients")
plt.ylabel("Distance Euclidienne")
plt.show()

# Appliquer l'Agglomerative Hierarchical Clustering avec le nombre de clusters souhaité
n_clusters = 3  # ajuster ce paramètre selon le dendrogramme
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(scaled_data)

# Ajouter les clusters aux données originales pour analyse
data['Cluster'] = clusters
print("Clustering terminé. Exemple des données avec clusters :")
print(data.head())
