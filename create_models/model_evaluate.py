from sklearn.metrics import accuracy_score, classification_report
from functools import wraps
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