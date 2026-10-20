import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.metrics import mean_poisson_deviance

df = pd.read_csv("data/primes_pures.csv")

# 1. Meme separation train/test que pour la validation
test = df.sample(frac=0.2, random_state=42)
train = df.drop(test.index)

# 2. GLM de frequence (3 variables), appris sur le train
variables = "C(Area) + C(DrivAge_classe) + C(BM_classe)"
m_glm = smf.glm("Nb ~ " + variables, data=train,
                family=sm.families.Poisson(),
                offset=np.log(train["Exposure"])).fit()
glm_pred = m_glm.predict(test, offset=np.log(test["Exposure"]))

# 3. XGBoost : variables numeriques, memes sinistres (Nb), exposition en offset
def preparer(d):
    X = d[["DrivAge", "BonusMalus", "VehAge", "VehPower", "Density"]].copy()
    X["Area"] = d["Area"].astype("category").cat.codes
    return X

dtrain = xgb.DMatrix(preparer(train), label=train["Nb"])
dtrain.set_base_margin(np.log(train["Exposure"].values))
dtest = xgb.DMatrix(preparer(test), label=test["Nb"])
dtest.set_base_margin(np.log(test["Exposure"].values))

params = {
    "objective": "count:poisson",
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "seed": 42,
}
m_xgb = xgb.train(params, dtrain, num_boost_round=300)
xgb_pred = m_xgb.predict(dtest)

# 4. Comparaison : deviance de Poisson sur le test (plus bas = meilleur)
y = test["Nb"]
dev_glm = mean_poisson_deviance(y, glm_pred)
dev_xgb = mean_poisson_deviance(y, xgb_pred)
print("Deviance moyenne GLM     :", round(dev_glm, 5))
print("Deviance moyenne XGBoost :", round(dev_xgb, 5))
print("Gain de XGBoost :", round(100 * (1 - dev_xgb / dev_glm), 2), "%")

# 5. Charge totale en nombre de sinistres
print()
print("Sinistres observes (test) :", int(y.sum()))
print("Sinistres predits GLM     :", round(glm_pred.sum()))
print("Sinistres predits XGBoost :", round(xgb_pred.sum()))