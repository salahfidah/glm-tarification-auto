import pandas as pd

freq = pd.read_csv("data/freMTPL2freq.csv")
sev = pd.read_csv("data/freMTPL2sev.csv")

print("=== TABLE FREQUENCE : 5 premieres lignes ===")
print(freq.head())

print()
print("=== TABLE SEVERITE : 5 premieres lignes ===")
print(sev.head())

print()
print("=== Colonnes de la table frequence ===")
print(freq.columns.tolist())