import pandas as pd

freq = pd.read_csv("data/freMTPL2freq.csv")

print("=== Dimensions ===")
print(freq.shape)

print()
print("=== Valeurs manquantes par colonne ===")
print(freq.isna().sum())

print()
print("=== Nombre de contrats selon le nombre de sinistres ===")
print(freq["ClaimNb"].value_counts().sort_index())

print()
print("=== Exposition totale (en annees) ===")
print(freq["Exposure"].sum())

print()
print("=== Frequence globale (sinistres par annee) ===")
print(freq["ClaimNb"].sum() / freq["Exposure"].sum())

print()
print("=== Statistiques descriptives ===")
print(freq[["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]].describe())