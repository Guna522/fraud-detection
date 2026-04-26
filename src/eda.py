import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/creditcard.csv")

print("Shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("\nClass distribution:\n", df["Class"].value_counts())

sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()