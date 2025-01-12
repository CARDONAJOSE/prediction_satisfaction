import unittest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        """
        chager le dataset et split le dataset en train et test 
        """
        # Load data
        data = pd.read_csv("./data/clean_data.csv")
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[['Class_Business','Seat comfort','Type of Travel_Personal Travel', 'Cleanliness','Online boarding','Class_Eco','Inflight entertainment', 'Type of Travel_Business travel']], data['Satisfaction'], test_size=0.2, random_state=False)

    def test_train_model(self):
        """ 
        entrainer le model et tester sa performance 
        """
        # Define parameters grid
        param_grid = {'n_neighbors': [3, 5, 7, 9, 10]}
        # Train model
        best_model_knn = train_model(self.X_train, self.y_train, param_grid)
        # Predict and evaluate
        y_pred_knn = best_model_knn.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred_knn)
        # Check accuracy
        self.assertGreater(accuracy, 0.5, "Model accuracy should be greater than 0.5")

    def test_save_model(self):
        # Train model
        best_model_knn = train_model(self.X_train, self.y_train, param_grid)
        # Save model
        model_path = './models/classification_knn_model.pkl'
        joblib.dump(best_model_knn, model_path)
        # Check if model is saved
        self.assertTrue(joblib.load(model_path) is not None, "Model should be saved successfully")

if __name__ == '__main__':
    unittest.main()