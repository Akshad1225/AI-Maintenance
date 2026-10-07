import pandas as pd

data = pd.read_csv("data/ai4i2020.csv")

print(data.head())
print(data.shape)
print(data.columns)
print(data["Machine failure"].value_counts())