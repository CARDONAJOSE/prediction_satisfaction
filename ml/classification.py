from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from outils import load_data
from train_evaluate import train_model, evaluate_model
from save import save_model
#from functools import wraps
import numpy as np
import pandas as pd
import mlflow
import sys
import os

# Ajoute le chemin parent au système
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# charge le dataset
x, y = load_data()

# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)

# definir la grille de parametres a tester
param_grid = {'n_neighbors': [3, 5, 7, 9, 10]}
 
print(12*"*")

# instanciation du model et entrainement    
model_knn = train_model(X_train, y_train, param_grid,KNeighborsClassifier() )

# prediction sur le jeu de test 
accuracy, raport, y_pred = evaluate_model(X_test, y_test, model_knn)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_params(raport)
mlflow.log_metric("precision", raport['macro avg']['precision'])

# Sauvegarde au chemin selon architecture 
save_model_knn = save_model(model_knn, filename='./models/model_knn.pkl')
mlflow.sklearn.log_model(model_knn, "model_knn")


# Matriz de Confusión
#   precision    recall  f1-score   support

#            0       0.87      0.93      0.90     14657
#            1       0.90      0.81      0.85     11319

#     accuracy                           0.88     25976
#    macro avg       0.88      0.87      0.88     25976
# weighted avg       0.88      0.88      0.88     25976