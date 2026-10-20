import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/donnees_preparees.csv")

# Découpage en classes
df["DrivAge_classe"] = pd.cut(
    df["DrivAge"], bins=[17, 21, 25, 30, 40, 50, 60, 70, 100]
).astype(str)
df["BM_classe"] = pd.cut(
    df["BonusMalus"], bins=[49, 50, 60, 70, 80, 100, 120, 150, 230]
).astype(str)


def frequence_par(colonne):
    t = df.groupby(colonne, sort=False).agg(
        sinistres=("ClaimNb", "sum"),
        exposition=("Exposure", "sum"),
    )
    t["frequence"] = t["sinistres"] / t["exposition"]
    return t


# Ordre correct des classes
ordre_age = sorted(df["DrivAge_classe"].unique(), key=lambda s: float(s.split(",")[0][1:]))
ordre_bm = sorted(df["BM_classe"].unique(), key=lambda s: float(s.split(",")[0][1:]))

graphiques = [
    ("Area", "Zone (A = rural, F = très urbain)", None),
    ("DrivAge_classe", "Âge du conducteur", ordre_age),
    ("BM_classe", "Bonus-malus", ordre_bm),
    ("VehGas", "Carburant", None),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, (col, titre, ordre) in zip(axes.flat, graphiques):
    t = frequence_par(col)
    if ordre is not None:
        t = t.loc[ordre]
    else:
        t = t.sort_index()
    ax.bar(t.index.astype(str), t["frequence"], color="steelblue")
    ax.axhline(
        df["ClaimNb"].sum() / df["Exposure"].sum(),
        color="red", linestyle="--", label="Fréquence globale",
    )
    ax.set_title("Fréquence par " + titre)
    ax.set_ylabel("Sinistres par année-risque")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

plt.tight_layout()
plt.savefig("outputs/frequence_par_variable.png", dpi=150)
print("Graphique enregistré : outputs/frequence_par_variable.png")
plt.show()