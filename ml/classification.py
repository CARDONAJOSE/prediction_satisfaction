from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from .outils import data_load, timer
import numpy as np
import pandas as pd
import mlflow
import joblib


# # charge le dataset
# data = pd.read_csv("./data/clean_data.csv")

# # split le dataset, separer les features et la target
# x = data[['Class_Business','Seat comfort','Type of Travel_Personal Travel', 'Cleanliness','Online boarding','Class_Eco','Inflight entertainment', 'Type of Travel_Business travel']]
# y = data['Satisfaction']
#data_load.load_data()
x, y = data_load.load_data()
# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)

# Démarrer une expérience MLflow
with mlflow.start_run():

    # definir la grille de parametres a tester
    param_grid = {'n_neighbors': [3, 5, 7, 9, 10]}
    # instanciation du model et entrainement
    @timer
    def train_model(X_train, y_train, param_grid):
        """
    entrainement du model 

    Parametres:
        X_train: the training data
        y_train: the training labels
        param_grid: the hyperparameters to test
    
    Returns:
        best_model_knn: meilleur model
        """
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Recuperer le meilleur model
        best_model_knn = grid_search.best_estimator_
        print(f"amelioration des hiperparámetres: {grid_search.best_params_}")
        return best_model_knn
    @timer
    def evaluate_model(best_model_knn):

        """
        Evaluation du model

        Parameters:
            best_model_knn: le meilleur model du grid search

        Returns:
            accuracy: la precision du model
        """
        y_pred_knn = best_model_knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_knn)
        print("Accuracy:", accuracy_score(y_test, y_pred_knn))
        print(classification_report(y_test, y_pred_knn))
        raport_knn = classification_report(y_test, y_pred_knn, output_dict=True)
        
        return accuracy, raport_knn
    
    print(12*"*")
    model_knn = train_model(X_train, y_train, param_grid)
    accuracy, raport = evaluate_model(train_model(X_train, y_train, param_grid))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params(raport)
    mlflow.log_metric("precision", raport['macro avg']['precision'])
    # Mesurer le temps d'entrainement
    #with timer.timeit("Entrainer le model"):
# predictions et evaluation du model

# Sauvegarde au chemin selon architecture
    @timer
    def save_model(best_model_knn):
        """
    Sauvegarde le modèle entraîné dans un fichier .pkl à l'emplacement
    ./models/classification_knn_model.pkl

    Parameters:
        best_model_knn: le modèle entraîné

    Returns:
        best_model_knn: le modèle sauvegardé
    """
        saved_model = joblib.dump(best_model_knn, './models/classification_knn_model.pkl')
        return saved_model
    
    mlflow.sklearn.log_model(model_knn, "model_knn")

    if __name__ == "__main__":
        save_model(model_knn)
        
# Matriz de Confusión
#   precision    recall  f1-score   support

#            0       0.87      0.93      0.90     14657
#            1       0.90      0.81      0.85     11319

#     accuracy                           0.88     25976
#    macro avg       0.88      0.87      0.88     25976
# weighted avg       0.88      0.88      0.88     25976