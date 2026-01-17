import pandas as pd

# Path to the .data file
path = "processed.cleveland.data"

# Column names from the UCI documentation
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Read the file
df = pd.read_csv(
    path,
    header=None,
    names=columns,
    na_values="?"
)

print(df.head())
