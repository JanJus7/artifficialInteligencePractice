import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv", header=0, sep=',', index_col=0)
df = df.astype(str)

ohe_df = pd.get_dummies(df, prefix=df.columns)

freq_items = apriori(ohe_df, min_support=0.005, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.8)
rules = rules.sort_values(by='confidence', ascending=False)

survived_rules = rules[rules['consequents'].astype(str).str.contains("Survived_Yes")]
not_survived_rules = rules[rules['consequents'].astype(str).str.contains("Survived_No")]

print("\nSurvived::")
print(survived_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

print("\nDeath:")
print(not_survived_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True)
plt.show()