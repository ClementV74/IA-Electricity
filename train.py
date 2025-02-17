from sklearn.ensemble import IsolationForest
import pandas as pd

# Charger les données
df = pd.read_csv("train_data_with_variations.csv")

# Séparer les features (tension, amperage) et la cible (label)
X = df[['Tension', 'Amperage_mA']]

# Entraîner le modèle IsolationForest pour détecter les anomalies
model = IsolationForest(n_estimators=100, contamination=0.1)  # La contamination est la proportion d'anomalies
model.fit(X)

# Sauvegarder le modèle
import joblib
joblib.dump(model, 'model_anomalie_variation.pkl')
