import pandas as pd

df = pd.read_csv("data/donnees_preparees.csv")
sev = pd.read_csv("data/freMTPL2sev.csv")

# 1. Statistiques sur les montants des sinistres
print("=== Montants des sinistres (table sev) ===")
print(sev["ClaimAmount"].describe().round(0))

# 2. Les 10 plus gros sinistres
print()
print("=== 10 plus gros sinistres ===")
print(sev.sort_values("ClaimAmount", ascending=False).head(10))

# 3. Part du total due aux 20 plus gros sinistres
top20 = sev["ClaimAmount"].nlargest(20).sum()
total = sev["ClaimAmount"].sum()
print()
print("Part des 20 plus gros sinistres dans le total :", round(100 * top20 / total, 1), "%")

# 4. Les plus gros sinistres, avec l'age du conducteur
gros = sev.merge(df[["IDpol", "DrivAge"]], on="IDpol", how="left")
print()
print("=== 10 plus gros sinistres avec l'age du conducteur ===")
print(gros.sort_values("ClaimAmount", ascending=False).head(10))