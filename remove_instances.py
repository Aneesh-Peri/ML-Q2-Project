import pandas as pd
df = pd.read_csv("heart_cleveland.csv", na_values=["?", ""])
df_clean = df.dropna(subset=["ca", "thal"])
df_clean.to_csv("heart_disease_clean.csv", index=False)
