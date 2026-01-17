import pandas as pd

df = pd.read_csv("heart_cleveland.csv")
df["target"] = df["target"].clip(upper=1)
df.to_csv("better_cleveland.csv", index=False)
