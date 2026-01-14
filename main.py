import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import json
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# import dataset
# fetch dataset 
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes-like object)
# estrai le feature e le target come DataFrame separati
features = heart_disease.data.features.copy()
targets = heart_disease.data.targets.copy()

# metadata
print("Metadata for dataset Heart Disease:") 
print(json.dumps(heart_disease.metadata, indent=2, default=str))
# variable information
print("Variable Information for Heart Disease:") 
print(heart_disease.variables)

# Visualizza le prime righe delle feature e delle target
print(features.head())
print(targets.head())

# Rimuovi le righe che contengono valori NaN nelle feature
# e riallinea le target sullo stesso indice
X = features.dropna()
y = targets.loc[X.index]
print("DataFrame dopo dropna, shape:", X.shape)
print("Targets dopo riallineamento, shape:", y.shape)

scaler = StandardScaler()
numerical_cols = ["age", "chol", "trestbps"]

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print(X.head())

X = pd.get_dummies(X, columns=["sex", "cp", "thal"], drop_first=True)
print(X.head())


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Shape before SMOTE:", X.shape, y.shape)
print("Shape after SMOTE:", X_resampled.shape, y_resampled.shape)
print(X_resampled.head())