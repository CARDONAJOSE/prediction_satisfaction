from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from functools import wraps
import time
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
@timer
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

@timer
def evaluate_model(X_test, y_test, best_model):

        """
        Evaluation du model

        Parameters:
            best_model_knn: le meilleur model du grid search
        
        Returns:
            accuracy: la precision du model
        """
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        raport = classification_report(y_test, y_pred, output_dict=True)
        
        return accuracy, raport, y_pred