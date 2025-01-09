from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

data = pd.read_csv("./data/clean_data.csv")
x= data.drop("class", axis=1)
y= data["class"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
