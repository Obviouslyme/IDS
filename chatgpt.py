# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Chargement des données
# Assurez-vous d'avoir un fichier CSV contenant vos données
data = pd.read_csv("votre_fichier.csv")

# Exploration des données (facultatif)
print(data.head())  # Affiche les premières lignes du jeu de données

# Séparation des caractéristiques (features) et de la cible (target)
X = data.drop('classe_intrusion', axis=1)  # features
y = data['classe_intrusion']  # target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
print(classification_report(y_test, y_pred))

# Vous pouvez également sauvegarder le modèle entraîné pour une utilisation ultérieure
# from joblib import dump
# dump(model, 'model.joblib')