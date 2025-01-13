from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from outils import load_data
from model_train import train_model
from model_evaluate import evaluate_model
from functools import wraps
import numpy as np
import pandas as pd
# import mlflow
import joblib
import mlflow
import time
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

# charge le dataset
x,y=load_data()

# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    # definir la grille de parametres a tester
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Inverso de la regularización
    'solver': ['liblinear', 'saga']  # Solvers para la regresión logística
    }
    # Instancier el modelo
    @timer
    model_logistic = train_model(X_train, y_train, param_grid,LogisticRegression())
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

    # prediction sur le jeu de test
    
    accuracy, raport, y_pred = evaluate_model(X_test, y_test, model_logistic)
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# save_path = "lineal_logistic_model.joblib"
# Sauvegarde au chemin selon architecture
#save_model_logistic = save_model(model_logistic, './models/lineal_logistic_model.pkl')
joblib.dump(model_logistic, '../models/lineal_logistic_model.pkl')