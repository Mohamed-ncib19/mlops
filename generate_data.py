import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Fixer la graine aléatoire pour garantir la reproductibilité.
# Les mêmes données seront générées à chaque exécution.
np.random.seed(42)

# Nombre de clients fictifs à générer.
N_CLIENTS = 500


# ============================================================
# 2. GÉNÉRATION DES DONNÉES CLIENTS
# ============================================================

# Chaque colonne représente une caractéristique (feature)
# susceptible d'être utilisée par un futur modèle de Machine Learning.

data = pd.DataFrame({
    "age": np.random.randint(18, 70, N_CLIENTS),
    "salaire": np.random.randint(1200, 8000, N_CLIENTS),
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"],
        N_CLIENTS
    ),
    "anciennete": np.random.randint(0, 15, N_CLIENTS),
    "consommation": np.random.randint(0, 100, N_CLIENTS)
})


# ============================================================
# 3. CRÉATION DE LA VARIABLE CIBLE : CHURN
# ============================================================

# Un client est considéré comme étant à risque de départ si :
# - sa consommation est inférieure à 30 ;
# - son ancienneté est inférieure à 2 ans.
#
# La variable cible prend deux valeurs :
# 1 = client à risque de départ
# 0 = client non considéré à risque selon cette règle.

conditions_churn = (
    (data["consommation"] < 30)
    & (data["anciennete"] < 2)
)

data["churn"] = conditions_churn.astype(int)


# ============================================================
# 4. EXPORTATION DES DONNÉES
# ============================================================

# Enregistrer le jeu de données dans un fichier CSV.
# index=False évite d'ajouter une colonne d'index inutile.

OUTPUT_FILE = "churn_data.csv"

data.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 5. VÉRIFICATION DES RÉSULTATS
# ============================================================

print("APERÇU DU JEU DE DONNÉES")
print(data.head())

print("\nINFORMATIONS GÉNÉRALES")
print(f"Nombre de clients : {len(data)}")
print(f"Nombre de variables : {len(data.columns) - 1}")

print("\nRÉPARTITION DU CHURN")
print(data["churn"].value_counts().rename(
    index={0: "Non à risque", 1: "À risque"}
))

taux_churn = data["churn"].mean()

print(f"\nTaux de churn simulé : {taux_churn:.1%}")
print(f"\nFichier exporté : {OUTPUT_FILE}")
