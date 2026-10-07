import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/tiny_maintenance_data.csv")

print(data)

X = data[["temperature", "vibration", "running_hours"]]
y = data["failed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual answers:", list(y_test))