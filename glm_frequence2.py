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

formule = "ClaimNb ~ C(Area) + C(DrivAge_classe) + C(BM_classe)"
offset = np.log(df["Exposure"])

# 2. Modele Poisson SANS le carburant
poisson = smf.glm(
    formula=formule, data=df, family=sm.families.Poisson(), offset=offset
).fit()

# 3. Mesure de la surdispersion : Pearson chi2 / degres de liberte
dispersion = poisson.pearson_chi2 / poisson.df_resid
print("Deviance Poisson :", round(poisson.deviance, 1))
print("AIC Poisson      :", round(poisson.aic, 1))
print("Dispersion (Pearson chi2 / ddl) :", round(dispersion, 3))

# 4. Quasi-Poisson : memes coefficients, mais erreurs standard corrigees
quasi = smf.glm(
    formula=formule, data=df, family=sm.families.Poisson(), offset=offset
).fit(scale="X2")

# 5. Comparaison des p-values
comparaison = pd.DataFrame(
    {
        "coef": poisson.params.round(4),
        "p_Poisson": poisson.pvalues.round(4),
        "p_QuasiPoisson": quasi.pvalues.round(4),
    }
)
print()
print(comparaison)