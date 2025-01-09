from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
# import mlflow
import joblib
import matplotlib.pyplot as plt

# charge le dataset
data = pd.read_csv("./data/clean_data.csv")

# split le dataset, separer les features et la target
x = data[['Class_Business','Seat comfort','Type of Travel_Personal Travel', 'Cleanliness','Online boarding','Class_Eco','Inflight entertainment', 'Type of Travel_Business travel']]
y = data['Satisfaction']

# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)

# definir la grille de parametres a tester
param_grid = {'n_neighbors': [3, 5, 7, 9, 10]}

# instanciation du model et entrainement
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Recuperer el mejor modelo
best_model_knn = grid_search.best_estimator_
print(f"Meilleurs hiperparámetres: {grid_search.best_params_}")


# predictions et evaluation du model
y_pred_knn = best_model_knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Sauvegarde au chemin selon architecture
joblib.dump(best_model_knn, './models/classification_knn_model.pkl')

#   precision    recall  f1-score   support

#            0       0.87      0.93      0.90     14657
#            1       0.90      0.81      0.85     11319

#     accuracy                           0.88     25976
#    macro avg       0.88      0.87      0.88     25976
# weighted avg       0.88      0.88      0.88     25976