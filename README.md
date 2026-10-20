# Tarification en assurance automobile : GLM Poisson × Gamma

Projet académique (Master Actuariat et Finance des Marchés, FSJES Aïn Sebaâ,
Université Hassan II de Casablanca).

Construction d'une prime pure individualisée selon le modèle collectif
(GLM de Poisson pour la fréquence, GLM Gamma pour le coût moyen) sur le jeu
de données freMTPL2 (678 013 contrats), avec comparaison à XGBoost.

## Résultats principaux
- Bonus-malus : facteur le plus discriminant pour la fréquence
- Surdispersion (≈ 2,9) corrigée par un quasi-Poisson
- Sinistres plafonnés à 100 000 € pour stabiliser le modèle de coût
- Prime pure moyenne : 138 € par année-risque
- Test (20 %) : charge sous-estimée de 6,6 %, classement des risques cohérent
- XGBoost : gain de seulement 1,3 % sur la déviance de Poisson

## Rapport complet
Voir [rapport_glm_tarification.pdf](rapport_glm_tarification.pdf).

## Installation
pip install pandas numpy statsmodels scikit-learn xgboost matplotlib

## Ordre d'exécution
1. telecharger_donnees.py
2. nettoyage.py
3. glm_frequence2.py
4. glm_cout_plafonne.py
5. prime_pure.py
6. validation.py
7. xgboost_comparaison.py

Les autres fichiers (explorer.py, analyse1-3.py, glm_frequence.py,
glm_cout.py, glm_cout2.py, verif_gros_sinistres.py) sont des étapes
exploratoires.

## Données
freMTPL2 (OpenML, identifiants 41214 et 41215), téléchargées par
telecharger_donnees.py.

## Auteur
Salah Fidah
