from sklearn.metrics import accuracy_score, classification_report

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