# ===== 1. Importer les bibliothèques =====
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ===== 2. Charger les données (une seule fois) =====
df = pd.read_csv("churn_data.csv")  # Tableau de 500 clients
y = df["churn"]  # 1 = le client part, 0 = le client reste


# ===== 3. Fonction : préparer les données pour une liste de colonnes =====
def preparer(colonnes):
    # Transformer les variables catégorielles en colonnes 0/1
    X = pd.get_dummies(df[colonnes], drop_first=True)

    # 80 % pour l'apprentissage, 20 % pour le test
    # random_state=42 garantit un découpage reproductible
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


# ===== 4. Fonction : lancer UN essai et l'enregistrer dans MLflow =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    # Ouvrir un run MLflow au nom de l'essai
    with mlflow.start_run(run_name=nom):

        # --- Création et entraînement du modèle ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,  # Nombre d'arbres
            max_depth=max_depth,        # Profondeur maximale des arbres
            random_state=42,            # Résultats reproductibles
        )

        model.fit(X_train, y_train)

        # --- Prédictions ---
        pred = model.predict(X_test)

        # --- Paramètres : réglages de cet essai ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # --- Métriques : évaluation du modèle ---
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test,
            pred,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            pred,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            pred,
            zero_division=0,
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # --- Artifact : sauvegarder le modèle dans ce run ---
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        # --- Afficher les résultats ---
        print(
            nom,
            "-> accuracy :", round(accuracy, 3),
            "| f1 :", round(f1, 3),
        )


# ===== 5. Programme principal =====

# Définir l'expérience MLflow
mlflow.set_experiment("churn-prediction")

# Toutes les colonnes utilisées pour l'apprentissage
TOUT = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation",
]

# Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
lancer_essai("essai-1", 100, 5, TOUT)

# Essai 2 : modèle plus simple, 50 arbres et profondeur 3
lancer_essai("essai-2", 50, 3, TOUT)

# Essai 3 : sans la colonne salaire
lancer_essai(
    "essai-3",
    100,
    5,
    [
        "age",
        "abonnement",
        "anciennete",
        "consommation",
    ],
)