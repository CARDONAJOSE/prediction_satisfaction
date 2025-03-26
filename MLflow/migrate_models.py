import os
import shutil
import mlflow
from datetime import datetime

def migrate_models():
    """Migrar modelos existentes a la nueva estructura"""
    # Crear directorios si no existen
    os.makedirs('prediction_satisfaction/MLflow/models', exist_ok=True)
    os.makedirs('prediction_satisfaction/MLflow/visualizations', exist_ok=True)
    
    # Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("migrated_satisfaction_prediction")
    
    # Mover modelos existentes
    model_files = {
        'model_knn.pkl': 'knn_model.pkl',
        'gradient_boost_model.pkl': 'gradient_boost_model.pkl',
        'lineal_logistic_model.pkl': 'logistic_regression_model.pkl',
        'classification_knn_model.pkl': 'knn_classification_model.pkl'
    }
    
    for old_name, new_name in model_files.items():
        old_path = f'prediction_satisfaction/models/{old_name}'
        new_path = f'prediction_satisfaction/MLflow/models/{new_name}'
        
        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
            print(f"Migrado: {old_name} -> {new_name}")
    
    # Registrar la migración en MLflow
    with mlflow.start_run(run_name="model_migration"):
        mlflow.log_param("migration_date", datetime.now().strftime("%Y-%m-%d"))
        mlflow.log_param("migrated_models", list(model_files.values()))
        
        # Registrar los modelos en MLflow
        for model_name in model_files.values():
            model_path = f'prediction_satisfaction/MLflow/models/{model_name}'
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, f"models/{model_name}")

def cleanup_old_files():
    """Limpiar archivos antiguos después de la migración"""
    # Directorios a limpiar
    dirs_to_clean = [
        'prediction_satisfaction/create_models',
        'prediction_satisfaction/models'
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Eliminado directorio: {dir_path}")

def main():
    print("Iniciando migración de modelos...")
    
    # Migrar modelos
    migrate_models()
    
    # Limpiar archivos antiguos
    cleanup_old_files()
    
    print("\nMigración completada. Los modelos ahora están en:")
    print("- prediction_satisfaction/MLflow/models/")
    print("- prediction_satisfaction/MLflow/visualizations/")
    
    print("\nPara usar los nuevos modelos, ejecuta:")
    print("python prediction_satisfaction/MLflow/improved_models.py")

if __name__ == "__main__":
    main() 