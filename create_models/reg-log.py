from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from outils import load_data
from train_evaluate import train_model, evaluate_model
from save import save_model
import numpy as np
import pandas as pd
import mlflow

# charge le dataset
x,y=load_data()

# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# definir la grille de parametres a tester
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Inverso de la regularización
    'solver': ['liblinear', 'saga']  # Solvers para la regresión logística
}

# Instancier el modelo
model_logistic = train_model(X_train, y_train, param_grid,LogisticRegression(max_iter=1000))

# prediction sur le jeu de test 
accuracy, raport, y_pred = evaluate_model(X_test, y_test, model_logistic)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_params(raport)
mlflow.log_metric("precision", raport['macro avg']['precision'])
# Sauvegarde au chemin selon architecture
save_model_logistic = save_model(model_logistic, './models/lineal_logistic_model.pkl')
mlflow.sklearn.log_model(model_logistic, "model_logistic")

# Matriz de Confusión     

#Accuracy: 0.839736680012319
#               precision    recall  f1-score   support

#            0       0.86      0.86      0.86     14622
#            1       0.81      0.82      0.82     11354

#     accuracy                           0.84     25976
#    macro avg       0.84      0.84      0.84     25976
# weighted avg       0.84      0.84      0.84     25976

# evaluate_model a pris 0.0811 secondes