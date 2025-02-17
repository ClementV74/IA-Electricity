import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# Paramètres de la simulation
n_samples = 100000
normal_variation = 100  # Variation normale autour de 10mA
anomaly_variation = 500  # Variation anormale (trop grande)
seuil_variation = 150  # Seuil au-delà duquel on considère la variation comme anormale

# Générer les données
np.random.seed(42)

# Liste pour stocker les données
data = []
labels = []  # 1 = normal, -1 = anormal

# Génération des données normales
for _ in range(n_samples // 2):
    tension = np.random.normal(230, 0.5)  # Variation de tension autour de 230V
    amperage = np.random.normal(1000, normal_variation)  # Amperage normal
    data.append([tension, amperage])
    labels.append(1)  # 1 pour normal

# Génération des anomalies
for _ in range(n_samples // 2):
    tension = np.random.normal(230, 0.5)
    amperage = np.random.normal(1000, anomaly_variation)  # Variation excessive
    data.append([tension, amperage])
    labels.append(-1)  # -1 pour anomalie

# Convertir en DataFrame
df = pd.DataFrame(data, columns=["Tension", "Amperage_mA"])
df["Label"] = labels

# Enregistrer les données dans un fichier CSV
df.to_csv("train_data_with_variations.csv", index=False)
