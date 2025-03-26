import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib
import os
from outils import load_data

# Configurar MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("improved_satisfaction_prediction")

# Crear directorio para modelos si no existe
os.makedirs('prediction_satisfaction/MLflow/models', exist_ok=True)

def save_model_pkl(model, model_name):
    """Guardar modelo en formato .pkl"""
    model_path = f'prediction_satisfaction/MLflow/models/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")

# def load_data():
#     """Cargar datos desde el archivo CSV"""
#     # Obtener la ruta absoluta del directorio actual
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # Construir la ruta al archivo CSV
#     csv_path = os.path.join(os.path.dirname(current_dir), 'data', 'clean_data.csv')
    
#     print(f"Cargando datos desde: {csv_path}")
#     df = pd.read_csv(csv_path)
    
#     # Eliminar la columna 'Unnamed: 0' y 'id' si existen
#     if 'Unnamed: 0' in df.columns:
#         df = df.drop('Unnamed: 0', axis=1)
#     if 'id' in df.columns:
#         df = df.drop('id', axis=1)
    
#     # Convertir la columna Satisfaction a numérico
#     df['Satisfaction'] = df['Satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})
    
#     # Verificar y mostrar valores nulos
#     null_counts = df.isnull().sum()
#     print("\nValores nulos por columna:")
#     for col, count in null_counts.items():
#         if count > 0:
#             print(f"- {col}: {count} valores nulos")
    
#     # Eliminar filas con valores nulos
#     df = df.dropna()
    
#     print(f"\nTamaño del DataFrame después de limpiar: {df.shape}")
#     print("\nColumnas disponibles:")
#     for col in df.columns:
#         print(f"- {col}")
    
#     return df

# def preprocess_data(df):
#     """Preprocesar los datos"""
#     # Separar características y objetivo
#     X = df.drop('Satisfaction', axis=1)
#     y = df['Satisfaction']
    
    # # Verificar que no hay valores nulos
    # if X.isnull().any().any() or y.isnull().any():
    #     print("¡Advertencia! Aún hay valores nulos en los datos")
    #     return None, None, None, None, None
    
    # # Dividir datos en entrenamiento y prueba
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # Escalar características
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

x, y = load_data()

X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=False)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Entrenar y evaluar el modelo"""
    if X_train is None or y_train is None:
        print(f"No se puede entrenar {model_name} debido a datos inválidos")
        return
        
    with mlflow.start_run(run_name=model_name):
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_mean", cv_mean)
        mlflow.log_metric("cv_std", cv_std)
        
        # Registrar el modelo en MLflow
        mlflow.sklearn.log_model(model, f"model_{model_name}")
        
        # Guardar el modelo en formato .pkl
        save_model_pkl(model, model_name)
        
        # Crear y guardar matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.savefig(f'../src/assets/confusion_matrix_{model_name}.png')
        plt.close()
        
        print(f"\nResultados para {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

def main():
    # Cargar datos
    print("Cargando datos...")
    x, y = load_data()
    
    # Preprocesar datos
    print("Preprocesando datos...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)
    
    # Definir modelos y sus parámetros
    models = {
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'gradient_boost': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    # Entrenar y evaluar cada modelo
    for name, model_info in models.items():
        print(f"\nEntrenando {name}...")
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        train_and_evaluate_model(grid_search, name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 