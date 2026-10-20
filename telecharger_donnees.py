from sklearn.datasets import fetch_openml
import os

os.makedirs("data", exist_ok=True)

print("Téléchargement de la table fréquence...")
freq = fetch_openml(data_id=41214, as_frame=True).frame
print("Téléchargement de la table sinistres...")
sev = fetch_openml(data_id=41215, as_frame=True).frame

freq.to_csv("data/freMTPL2freq.csv", index=False)
sev.to_csv("data/freMTPL2sev.csv", index=False)

print("Fréquence :", freq.shape)
print("Sévérité  :", sev.shape)