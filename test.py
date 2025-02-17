import warnings
import joblib
import pandas as pd

# Ignorer les avertissements spécifiques pour ce script
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# Charger le modèle
model = joblib.load('model_anomalie_variation.pkl')

# Simuler une série de nouvelles données avec variations
nouvelles_donnees = pd.DataFrame([
    [230.0, 1000.0],  # Valeur normale
    [230.0, 1200.0],  # Légère variation (normale)
    [230.0, 2000.0],  # Variation trop importante (anormale)
    [230.0, 5000.0],  # Variation trop importante (anormale)
], columns=["Tension", "Amperage_mA"])

# Variables pour suivre les anomalies
anomalies_detectees = 0
seuil_anomalies = 2  # Seuil d'anomalies avant de prendre une décision
temps_anomalie = 0  # Durée de persistance des anomalies

# Prédire avec le modèle et prendre des décisions
for index, row in nouvelles_donnees.iterrows():
    prediction = model.predict([row])  # Passer les données sous forme de liste de caractéristiques
    
    if prediction == -1:
        print(f"Anomalie détectée : Tension = {row['Tension']}V, Amperage = {row['Amperage_mA']}mA")
        anomalies_detectees += 1
        temps_anomalie += 1  # Compter le temps d'anomalie
    else:
        print(f"Normal : Tension = {row['Tension']}V, Amperage = {row['Amperage_mA']}mA")
        # Si les anomalies ne persistent pas, reset du compteur
        if anomalies_detectees > 0:
            anomalies_detectees -= 1
            temps_anomalie = 0

    # Prendre une décision basée sur la persistance des anomalies
    if anomalies_detectees >= seuil_anomalies:
        print("Anomalie persistante détectée. Action requise : Arrêter le système.")
        break
    elif temps_anomalie > 2:
        print("Anomalie détectée pendant un certain temps, nécessite une intervention.")
    else:
        print("Système stable.")
