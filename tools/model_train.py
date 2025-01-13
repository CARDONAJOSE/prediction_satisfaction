from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def train_model(X_train, y_train, param_grid,model_type):
        """
    entrainement du model 

    Parametres:
        X_train: the training data
        y_train: the training labels
        param_grid: the hyperparameters to test
    
    Returns:
        best_model_knn: meilleur model
        """
        grid_search = GridSearchCV(model_type, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Recuperer le meilleur model
        best_model = grid_search.best_estimator_
        print(f"amelioration des hiperparámetres: {grid_search.best_params_}")
        return best_model

