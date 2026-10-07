import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

data = pd.read_csv("data/ai4i2020.csv")

data["Usage_Score"] = data["Rotational speed [rpm]"] + data["Torque [Nm]"] + data["Tool wear [min]"]

data["Criticality"] = pd.qcut(data["Usage_Score"], q=3, labels=["Low", "Medium", "High"])

print(data["Criticality"].value_counts())

X = data[["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]]
y = data["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(class_weight="balanced")

scores = cross_val_score(model, X, y, cv=5, scoring="recall")
print("Cross-validated recall scores:", scores)
print("Average recall:", scores.mean())

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print(classification_report(y_test, predictions))

data["Risk_Prediction"] = model.predict(X)

priority_map = {"Low": 1, "Medium": 2, "High": 3}
data["Criticality_Score"] = data["Criticality"].astype(str).map(priority_map)

data["Priority_Score"] = data["Risk_Prediction"] * data["Criticality_Score"]

def assign_schedule(score):
    if score == 3:
        return "Immediate (within 3 days)"
    elif score == 2:
        return "Short-term (within 2 weeks)"
    elif score == 1:
        return "Routine (within 1 month)"
    else:
        return "No action needed"

def assign_resources(priority_score):
    if priority_score == 3:
        return "2 Technicians, Spare Parts: In Stock Priority"
    elif priority_score == 2:
        return "1 Technician, Spare Parts: Standard Order"
    elif priority_score == 1:
        return "1 Technician, Spare Parts: Routine Check"
    else:
        return "No resources needed"

data["Maintenance_Schedule"] = data["Priority_Score"].apply(assign_schedule)
data["Resource_Plan"] = data["Priority_Score"].apply(assign_resources)

print(data[["UDI", "Priority_Score", "Maintenance_Schedule", "Resource_Plan"]].sort_values(by="Priority_Score", ascending=False).head(10))