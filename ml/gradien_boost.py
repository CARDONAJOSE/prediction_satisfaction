from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# import mlflow.sklearn
import numpy as np
import pandas as pd
# import mlflow
import joblib
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# charge le dataset
data = pd.read_csv("./data/clean_data.csv")

# split le dataset, separer les features et la target
x = data[['Class_Business','Seat comfort','Type of Travel_Personal Travel', 'Cleanliness','Online boarding','Class_Eco','Inflight entertainment', 'Type of Travel_Business travel']]
y = data['Satisfaction']

# ['Gender_Female', 'Gender_Male', 'Customer Type_Loyal Customer',
#        'Customer Type_disloyal Customer',
#        'Class_Eco Plus', 'id', 'Age', 'Flight Distance',
#        'Inflight wifi service', 'Departure/Arrival time convenient',
#        'Ease of Online booking', 'Gate location', 'Food and drink',
#        'On-board service', 'Leg room service', 'Baggage handling',
#        'Checkin service', 'Inflight service',
#        'Departure Delay in Minutes', 'Arrival Delay in Minutes',
#        'Satisfaction'],

# split le dataset en train et test 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)

# definir la grille de parametres a tester

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [1, 3, 5]
}

# Instancier el modelo
model_gb = GradientBoostingClassifier(random_state=42)

# Configurer GridSearchCV 
grid_search = GridSearchCV(estimator=model_gb, param_grid=param_grid, cv=5, scoring='accuracy')

# Entrenar el modelo con GridSearchCV
grid_search.fit(X_train, y_train)

# choisir le meilleur modele selon accuracy
best_model = grid_search.best_estimator_
print("Mejores hiperparámetros:", grid_search.best_params_)

# prediction sur le jeu de test 
y_pred_gb = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

# Sauvegarde au chemin selon architecture
joblib.dump(model_gb, './models/gradient_boost_model.pkl')

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_gb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión: gradient_boost")

# Sauvegarde de la Matriz de Confusión au chemin selon architecture
plt.savefig('./src/assets/confusion_matrix_gradient_boost.png') 

# Precision, Recall et F1-Score
precision_gb = precision_score(y_test, y_pred_gb, average='weighted')
print("Precision:", precision_gb)

recall_gb = recall_score(y_test, y_pred_gb, average='weighted')
print("Recall:", recall_gb)

f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
print("F1-Score:", f1_gb)

#               precision    recall  f1-score   support

#            0       0.87      0.89      0.88     14657
#            1       0.85      0.83      0.84     11319

#     accuracy                           0.86     25976
#    macro avg       0.86      0.86      0.86     25976
# weighted avg       0.86      0.86      0.86     25976
