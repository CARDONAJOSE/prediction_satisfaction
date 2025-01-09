from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import mlflow.sklearn
import numpy as np
import pandas as pd
# import mlflow
import joblib
from sklearn.metrics import accuracy_score, classification_report

# charge le dataset
data = pd.read_csv("../data/clean_data.csv")

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

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save_path = "lineal_logistic_model.joblib"
# Sauvegarde au chemin selon architecture
joblib.dump('../models/lineal_logistic_model.pkl')