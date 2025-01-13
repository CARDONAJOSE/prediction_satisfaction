from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from outils import load_data
from model_train import train_model
from model_evaluate import evaluate_model
from functools import wraps
#from ..tools import save_model
import time
import numpy as np
import pandas as pd
import mlflow
import joblib
import sys
import os

# Ajoute le chemin parent au système
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Décorateur pour mesurer le temps d'exécution
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = time.perf_counter()
        resultat = func(*args, **kwargs)
        fin = time.perf_counter()
        duree = fin - debut
        print(f"{func.__name__} a pris {duree:.4f} secondes")
        return resultat
    return wrapper

x, y = load_data()
# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)

# Démarrer une expérience MLflow
with mlflow.start_run():

    # definir la grille de parametres a tester
    param_grid = {'n_neighbors': [3, 5, 7, 9, 10]}
    # instanciation du model et entrainement
 
    print(12*"*")
    
    model_knn = train_model(X_train, y_train, param_grid)
    accuracy, raport, y_pred = evaluate_model(train_model(X_train, y_train, param_grid))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params(raport)
    mlflow.log_metric("precision", raport['macro avg']['precision'])

    # Mesurer le temps d'entrainement
    #with timer.timeit("Entrainer le model"):
# predictions et evaluation du model

# Sauvegarde au chemin selon architecture
    
    # def save_model(best_model_knn):
    #     """
    # Sauvegarde le modèle entraîné dans un fichier .pkl à l'emplacement
    # ./models/classification_knn_model.pkl

    # Parameters:
    #     best_model_knn: le modèle entraîné

    # Returns:
    #     best_model_knn: le modèle sauvegardé
    # """
    #     saved_model = joblib.dump(best_model_knn, './models/classification_knn_model.pkl')
    #     return saved_model
    
    #save_model_knn = save_model(model_knn)
    mlflow.sklearn.log_model(model_knn, "model_knn")

    # if __name__ == "__main__":
    #     save_model(model_knn)

# Matriz de Confusión
#   precision    recall  f1-score   support

#            0       0.87      0.93      0.90     14657
#            1       0.90      0.81      0.85     11319

#     accuracy                           0.88     25976
#    macro avg       0.88      0.87      0.88     25976
# weighted avg       0.88      0.88      0.88     25976