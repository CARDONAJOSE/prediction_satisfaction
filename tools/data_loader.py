import os
import pandas as pd

def load_data():
    """
    Load the data from the clean_data.csv file and return it as a pandas DataFrame.
    """
    try:
        file_path = os.path.join(os.path.dirname(__file__), "../data/clean_data.csv")
        data = pd.read_csv(file_path)
        y = data['Satisfaction']
        x = data[['Class_Business', 'Seat comfort', 'Type of Travel_Personal Travel',
                  'Cleanliness', 'Online boarding', 'Class_Eco',
                  'Inflight entertainment', 'Type of Travel_Business travel']]
        return x, y
    except FileNotFoundError:
        raise FileNotFoundError("Le fichier 'clean_data.csv' est introuvable. Vérifiez le chemin.")
    except KeyError as e:
        raise KeyError(f"Colonnes manquantes dans le fichier CSV : {e}")
