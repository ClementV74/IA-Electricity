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
    [230.0, 1000.0],  # Variation trop importante (anormale)
    [230.0, 1500.0],
    [230.0, 1500.0], 
    [230.0, 1500.0],   
    [230.0, 11700.0],  
    [230.0, 11700.0], 
    [230.0, 11700.0],   
], columns=["Tension", "Amperage_mA"])

# Variables pour suivre les anomalies
anomalies_detectees = 0
seuil_anomalies = 3  # Seuil d'anomalies avant de prendre une décision (3 itérations)
temps_anomalie = 0  # Durée de persistance des anomalies
amperage_normal = 1000  # Valeur attendue en mA
tolérance_amperage = 150  # Variabilité acceptable (par exemple ±15% de la valeur attendue)

# Prédire avec le modèle et prendre des décisions
for index, row in nouvelles_donnees.iterrows():
    prediction = model.predict([row])  # Passer les données sous forme de liste de caractéristiques
    
    # Vérifier si l'anomalie est dans la tolérance
    if prediction == -1 and not (amperage_normal - tolérance_amperage <= row['Amperage_mA'] <= amperage_normal + tolérance_amperage):
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
    elif temps_anomalie > 1:  # 1 itération avec anomalie, pas encore assez pour actionner une alerte
        print("Anomalie détectée pendant un certain temps, nécessite une intervention.")
    else:
        print("Système stable.")
