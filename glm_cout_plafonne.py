import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

freq = pd.read_csv("data/donnees_preparees.csv")
sev = pd.read_csv("data/freMTPL2sev.csv")

# Table des contrats (sans la colonne ClaimAmount deja agregee)
contrats = freq.drop(columns=["ClaimAmount"])

# Preparer les variables
contrats["DrivAge_classe"] = pd.cut(contrats["DrivAge"], bins=[17, 21, 25, 30, 40, 50, 60, 70, 100]).astype(str)
contrats["BM_classe"] = pd.cut(contrats["BonusMalus"], bins=[49, 50, 60, 70, 80, 100, 120, 150, 230]).astype(str)
contrats["Area"] = contrats["Area"].astype(str).str.replace("'", "", regex=False)

formule = "CoutMoyen ~ C(Area) + C(DrivAge_classe) + C(BM_classe)"
famille = sm.families.Gamma(link=sm.families.links.Log())

# On teste deux plafonds : chaque sinistre est limite a ce seuil
for plafond in [50000, 100000]:
    print()
    print("==============================")
    print("PLAFOND PAR SINISTRE :", plafond, "euros")
    print("==============================")

    s = sev.copy()
    s["Montant"] = s["ClaimAmount"].clip(upper=plafond)
    agg = s.groupby("IDpol", as_index=False).agg(Total=("Montant", "sum"), Nb=("Montant", "count"))

    d = contrats.merge(agg, on="IDpol", how="inner")
    d["CoutMoyen"] = d["Total"] / d["Nb"]

    print("Nombre de contrats avec sinistre :", len(d))
    print("Cout moyen global :", round(d["Total"].sum() / d["Nb"].sum(), 2))

    modele = smf.glm(formula=formule, data=d, family=famille, freq_weights=d["Nb"]).fit()

    resultat = pd.DataFrame({
        "relativite": np.exp(modele.params).round(3),
        "p_value": modele.pvalues.round(3),
    })
    print(resultat)