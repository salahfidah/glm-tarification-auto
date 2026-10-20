import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("data/donnees_preparees.csv")

# 1. Garder uniquement les contrats qui ont eu au moins un sinistre
sin = df[(df["ClaimNb"] > 0) & (df["ClaimAmount"] > 0)].copy()
print("Nombre de contrats avec sinistre :", len(sin))

# 2. Cout moyen par sinistre pour chaque contrat
sin["CoutMoyen"] = sin["ClaimAmount"] / sin["ClaimNb"]
print("Cout moyen global :", round(sin["CoutMoyen"].mean(), 2))

# 3. Preparer les variables
sin["DrivAge_classe"] = pd.cut(sin["DrivAge"], bins=[17, 21, 25, 30, 40, 50, 60, 70, 100]).astype(str)
sin["BM_classe"] = pd.cut(sin["BonusMalus"], bins=[49, 50, 60, 70, 80, 100, 120, 150, 230]).astype(str)
sin["Area"] = sin["Area"].astype(str).str.replace("'", "", regex=False)

# 4. GLM Gamma avec lien log
formule = "CoutMoyen ~ C(Area) + C(DrivAge_classe) + C(BM_classe)"
famille = sm.families.Gamma(link=sm.families.links.Log())
modele = smf.glm(formula=formule, data=sin, family=famille, freq_weights=sin["ClaimNb"]).fit()

print(modele.summary())
print()
print("=== Relativites (exp des coefficients) ===")
print(np.exp(modele.params).round(3))