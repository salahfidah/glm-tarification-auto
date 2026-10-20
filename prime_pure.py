import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("data/donnees_preparees.csv").drop(columns=["ClaimAmount"])
sev = pd.read_csv("data/freMTPL2sev.csv")

# Montants plafonnes a 100 000 euros, agreges par contrat
s = sev.copy()
s["Montant"] = s["ClaimAmount"].clip(upper=100000)
agg = s.groupby("IDpol", as_index=False).agg(Total=("Montant", "sum"), Nb=("Montant", "count"))

# Nombre de sinistres = nombre de lignes dans la table des montants
df = df.merge(agg, on="IDpol", how="left")
df["Nb"] = df["Nb"].fillna(0)
df["Total"] = df["Total"].fillna(0)
print("Sinistres (ClaimNb)     :", int(df["ClaimNb"].sum()))
print("Sinistres avec montant  :", int(df["Nb"].sum()))

df["DrivAge_classe"] = pd.cut(df["DrivAge"], bins=[17, 21, 25, 30, 40, 50, 60, 70, 100]).astype(str)
df["BM_classe"] = pd.cut(df["BonusMalus"], bins=[49, 50, 60, 70, 80, 100, 120, 150, 230]).astype(str)
df["Area"] = df["Area"].astype(str).str.replace("'", "", regex=False)

variables = "C(Area) + C(DrivAge_classe) + C(BM_classe)"

# 1. Frequence : on utilise Nb au lieu de ClaimNb
m_freq = smf.glm("Nb ~ " + variables, data=df,
                 family=sm.families.Poisson(), offset=np.log(df["Exposure"])).fit()

# 2. Cout moyen (Gamma)
d = df[df["Nb"] > 0].copy()
d["CoutMoyen"] = d["Total"] / d["Nb"]
m_cout = smf.glm("CoutMoyen ~ " + variables, data=d,
                 family=sm.families.Gamma(link=sm.families.links.Log()),
                 freq_weights=d["Nb"]).fit()

# 3. Predictions sur une annee complete
df_un_an = df.copy()
df_un_an["Exposure"] = 1.0
df["freq_pred"] = m_freq.predict(df_un_an, offset=np.zeros(len(df_un_an)))
df["cout_pred"] = m_cout.predict(df)
df["prime_pure"] = df["freq_pred"] * df["cout_pred"]

# 4. Verification globale
total_obs = df["Total"].sum()
total_pred = (df["prime_pure"] * df["Exposure"]).sum()
print()
print("Charge observee (plafonnee)  :", round(total_obs))
print("Charge predite par le modele :", round(total_pred))
print("Ecart :", round(100 * (total_pred / total_obs - 1), 1), "%")
print("Prime pure moyenne par annee-risque :", round(total_pred / df["Exposure"].sum(), 1), "euros")
print("Prime pure minimale :", round(df["prime_pure"].min(), 1))
print("Prime pure maximale :", round(df["prime_pure"].max(), 1))

df.to_csv("data/primes_pures.csv", index=False)