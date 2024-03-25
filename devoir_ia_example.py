# Import des library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

from sklearn import preprocessing

import numpy as np
import pandas as pd

#Import des sets
data_train = pd.read_csv('UNSW_NB15_training-set.csv', delimiter=',')
data_test = pd.read_csv('UNSW_NB15_testing-set.csv', delimiter=',')

training_nparray = data_train.to_numpy()
testing_nparray = data_test.to_numpy()

# Preprocess pour les data de train
enc = preprocessing.OrdinalEncoder()

encoded_dataset = enc.fit_transform(training_nparray)  # Toutes les catégories deviennent numériques
X_train = encoded_dataset[:, :-1]  # Toutes les lignes, mais on retire la dernière colonne
y_train = np.ravel(encoded_dataset[:, -1:])

# Preprocess pour les data de test
encoded_dataset = enc.fit_transform(testing_nparray)
X_test = encoded_dataset[:, :-1]
y_test = np.ravel(encoded_dataset[:, -1:])

# Algorithme 1 : KPPV (à 5 voisins)
kppv_model = KNeighborsClassifier(n_neighbors=5)
kppv_model.fit(X_train, y_train)
kppv_predictions = kppv_model.predict(X_test)

# Algorithme 2 : Arbre de décision
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# Algorithme 3 : Forêts aléatoires
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Algorithme 4 : Classificateur de Bayes
bayes_model = GaussianNB()
bayes_model.fit(X_train, y_train)
bayes_predictions = bayes_model.predict(X_test)

# Algorithme 5 : SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Algorithme 6 : Réseaux de neurones
rn_model = MLPClassifier()
rn_model.fit(X_train, y_train)
rn_predictions = rn_model.predict(X_test)

#On liste nos différents résultats en fonction de leur modèle
models = {
    'KPPV': kppv_predictions,
    'Arbre de décision': tree_predictions,
    'Forêts aléatoires': rf_predictions,
    'Classificateur de Bayes': bayes_predictions,
    'SVM': svm_predictions,
    'Réseaux de neurones': rn_predictions
}

#Affichage de toutes les données
for name, predictions in models.items():
    print(f"\nRapport de Classification pour {name}:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")