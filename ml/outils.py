import pandas as pd
from functools import wraps
import time

def load_data():
    """
    Load the data from the clean_data.csv file and return it as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv("./data/clean_data.csv")
    y = data['Satisfaction']
    x = data[['Class_Business','Seat comfort','Type of Travel_Personal Travel', 'Cleanliness','Online boarding','Class_Eco','Inflight entertainment', 'Type of Travel_Business travel']]    
    return x, y

# Décorateur pour mesurer le temps d'exécution
def timer(func):
    """ Décorateur pour mesurer le temps d'exécution d'une fonction. 
    
    Ce décorateur prend une fonction en argument et renvoie une fonction 
    wrapper qui appelle la fonction originale et mesure le temps d'exécution. 
    Il affiche le temps d'exécution dans la console.
    
    Parameters
    ----------
    func : function
        La fonction à mesurer.
    
    Returns
    -------
    wrapper : function
        La fonction wrapper qui appelle la fonction originale et mesure le 
        temps d'exécution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Méthode wrapper qui appelle la fonction func et mesura le temps
        d'exécution. Affiche le temps d'exécution dans la console."""
        debut = time.perf_counter()
        resultat = func(*args, **kwargs)
        fin = time.perf_counter()
        duree = fin - debut
        print(f"{func.__name__} a pris {duree:.4f} secondes")
        return resultat
    return wrapper

if __name__ == "__main__": 
    x, y = load_data()
    timed_load_data = timer(load_data)
    x, y = timed_load_data()
      