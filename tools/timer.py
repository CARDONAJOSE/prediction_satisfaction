import time
from functools import wraps

def timer(func):
    """ Décorateur pour mesurer le temps d'exécution d'une fonction. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = time.perf_counter()
        try:
            resultat = func(*args, **kwargs)
        except Exception as e:
            print(f"Erreur dans la fonction {func.__name__} : {e}")
            raise
        finally:
            fin = time.perf_counter()
            duree = fin - debut
            print(f"{func.__name__} a pris {duree:.4f} secondes")
        return resultat
    return wrapper
