from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Définir le chemin vers le fichier CSV et charger les données
file_path = os.path.join("E-commerce Customer behavior.csv")
data = pd.read_csv(file_path)
print("Données importées avec succès!!!")

# Afficher les premières lignes des données
print(data.head())

# Sélectionner les colonnes pertinentes et supprimer les lignes manquantes
data = data[['Total Spend', 'Items Purchased', 'Average Rating', 'Satisfaction Level']].dropna()
print(f"Données nettoyées : {data.shape[0]} lignes restantes.")

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Total Spend', 'Items Purchased', 'Average Rating']])
print("Données normalisées.")

# Appliquer le clustering Mean-Shift
mean_shifter = MeanShift(bandwidth=1)  # Définit la largeur de bande
clusters = mean_shifter.fit_predict(scaled_data)

# Récupérer les centres des clusters
centers = mean_shifter.cluster_centers_

# Affichage des résultats
x = scaled_data[:, 0]
y = scaled_data[:, 1]
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=clusters, s=50, cmap='viridis')
ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='+', label='Centers')
ax.set_xlabel('Total Spend')
ax.set_ylabel('Items Purchased')
plt.colorbar(scatter)
plt.legend()
plt.title("Mean-Shift Clustering")
plt.show()

# Sauvegarder la figure
fig.savefig("mean_shift_result.png")


