from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from ml.train_evaluate import train_model
from model_evaluate import evaluate_model
from functools import wraps
from outils import load_data
from ml.train_evaluate import train_model
from model_evaluate import evaluate_model
import mlflow
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sys
import os

# Ajoute le chemin parent au système
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

x, y = load_data()

# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)

# definir la grille de parametres a tester

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [1, 3, 5]
}

# Instancier el modelo
model_gb=train_model(X_train, y_train, param_grid,GradientBoostingClassifier())

# prediction sur le jeu de test 
accuracy, raport, y_pred = evaluate_model(X_test, y_test, model_gb)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_params(raport)
mlflow.log_metric("precision", raport['macro avg']['precision'])
    
# Sauvegarde au chemin selon architecture
joblib.dump(model_gb, './models/gradient_boost_model.pkl')

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión: gradient_boost")

# Sauvegarde de la Matriz de Confusión au chemin selon architecture
plt.savefig('./src/assets/confusion_matrix_gradient_boost.png') 

# Precision, Recall et F1-Score
precision_gb = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision_gb)

recall_gb = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall_gb)

f1_gb = f1_score(y_test, y_pred, average='weighted')
print("F1-Score:", f1_gb)

#               precision    recall  f1-score   support

#            0       0.87      0.89      0.88     14657
#            1       0.85      0.83      0.84     11319

#     accuracy                           0.86     25976
#    macro avg       0.86      0.86      0.86     25976
# weighted avg       0.86      0.86      0.86     25976
