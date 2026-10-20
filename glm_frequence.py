import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("data/donnees_preparees.csv")

# 1. Preparer les variables
df["DrivAge_classe"] = pd.cut(
    df["DrivAge"], bins=[17, 21, 25, 30, 40, 50, 60, 70, 100]
).astype(str)
df["BM_classe"] = pd.cut(
    df["BonusMalus"], bins=[49, 50, 60, 70, 80, 100, 120, 150, 230]
).astype(str)
df["Area"] = df["Area"].astype(str).str.replace("'", "", regex=False)
df["VehGas"] = df["VehGas"].astype(str).str.replace("'", "", regex=False)

# 2. Modele GLM de Poisson avec offset log(Exposure)
modele = smf.glm(
    formula="ClaimNb ~ C(Area) + C(DrivAge_classe) + C(BM_classe) + C(VehGas)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["Exposure"]),
).fit()

# 3. Resultats complets
print(modele.summary())

# 4. Relativites tarifaires (exponentielle des coefficients)
print()
print("=== Relativites tarifaires (exp des coefficients) ===")
print(np.exp(modele.params).round(3))