import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("data/primes_pures.csv")

# 1. Separer : 80 % pour apprendre (train), 20 % pour tester (test)
test = df.sample(frac=0.2, random_state=42)
train = df.drop(test.index)
print("Contrats train :", len(train))
print("Contrats test  :", len(test))

variables = "C(Area) + C(DrivAge_classe) + C(BM_classe)"

# 2. Modele de frequence, appris sur le train seulement
m_freq = smf.glm("Nb ~ " + variables, data=train,
                 family=sm.families.Poisson(),
                 offset=np.log(train["Exposure"])).fit()

# 3. Modele de cout moyen, appris sur le train seulement
t = train[train["Nb"] > 0].copy()
t["CoutMoyen"] = t["Total"] / t["Nb"]
m_cout = smf.glm("CoutMoyen ~ " + variables, data=t,
                 family=sm.families.Gamma(link=sm.families.links.Log()),
                 freq_weights=t["Nb"]).fit()

# 4. Predictions sur le test (jamais vu par le modele)
test = test.copy()
un_an = test.copy()
un_an["Exposure"] = 1.0
test["freq_pred"] = m_freq.predict(un_an, offset=np.zeros(len(un_an)))
test["cout_pred"] = m_cout.predict(test)
test["prime_pred"] = test["freq_pred"] * test["cout_pred"]

# 5. Comparaison globale : charge predite vs charge observee
obs = test["Total"].sum()
pred = (test["prime_pred"] * test["Exposure"]).sum()
print()
print("Charge observee sur le test :", round(obs))
print("Charge predite sur le test  :", round(pred))
print("Ecart :", round(100 * (pred / obs - 1), 1), "%")

# 6. Comparaison par decile de risque predit
test["decile"] = pd.qcut(test["prime_pred"], 10, labels=False, duplicates="drop") + 1
tab = test.groupby("decile").apply(
    lambda g: pd.Series({
        "contrats": len(g),
        "observe": g["Total"].sum(),
        "predit": (g["prime_pred"] * g["Exposure"]).sum(),
    })
)
tab["ratio_obs_pred"] = (tab["observe"] / tab["predit"]).round(2)
print()
print(tab.round(0))