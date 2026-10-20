import pandas as pd

freq = pd.read_csv("data/freMTPL2freq.csv")
sev = pd.read_csv("data/freMTPL2sev.csv")

# 1. Combien de valeurs sont concernees avant correction ?
print("Contrats avec Exposure > 1 :", (freq["Exposure"] > 1).sum())
print("Contrats avec ClaimNb > 4 :", (freq["ClaimNb"] > 4).sum())

# 2. Plafonner les valeurs anormales
freq["Exposure"] = freq["Exposure"].clip(upper=1)
freq["ClaimNb"] = freq["ClaimNb"].clip(upper=4)

# 3. Additionner les montants des sinistres de chaque contrat
sev_agg = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()

# 4. Relier les deux tables grace a IDpol
df = freq.merge(sev_agg, on="IDpol", how="left")
df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

# 5. Verification apres correction
print()
print("Exposure max apres plafonnement :", df["Exposure"].max())
print("ClaimNb max apres plafonnement :", df["ClaimNb"].max())
print("Dimensions de la table finale :", df.shape)
print("Montant total des sinistres :", df["ClaimAmount"].sum())

# 6. Sauvegarder la table preparee
df.to_csv("data/donnees_preparees.csv", index=False)
print()
print("Fichier data/donnees_preparees.csv cree.")