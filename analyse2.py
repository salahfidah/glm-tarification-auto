import pandas as pd

df = pd.read_csv("data/donnees_preparees.csv")

# Fréquence globale de référence
freq_globale = df["ClaimNb"].sum() / df["Exposure"].sum()
print("Fréquence globale :", round(freq_globale, 4))


def frequence_par(colonne):
    """Calcule la fréquence (sinistres par année-risque) pour chaque valeur d'une colonne."""
    t = df.groupby(colonne).agg(
        contrats=("IDpol", "count"),
        sinistres=("ClaimNb", "sum"),
        exposition=("Exposure", "sum"),
    )
    t["frequence"] = (t["sinistres"] / t["exposition"]).round(4)
    return t


# 1. Fréquence par zone (A = rural ... F = très urbain)
print()
print("=== Fréquence par Area ===")
print(frequence_par("Area"))

# 2. Fréquence par carburant
print()
print("=== Fréquence par VehGas ===")
print(frequence_par("VehGas"))

# 3. Fréquence par classes d'âge du conducteur
df["DrivAge_classe"] = pd.cut(
    df["DrivAge"],
    bins=[17, 21, 25, 30, 40, 50, 60, 70, 100],
)
print()
print("=== Fréquence par classe d'âge du conducteur ===")
print(frequence_par("DrivAge_classe"))

# 4. Fréquence par classes de bonus-malus
df["BM_classe"] = pd.cut(
    df["BonusMalus"],
    bins=[49, 50, 60, 70, 80, 100, 120, 150, 230],
)
print()
print("=== Fréquence par classe de bonus-malus ===")
print(frequence_par("BM_classe"))