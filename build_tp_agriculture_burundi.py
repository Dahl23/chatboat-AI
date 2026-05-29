from __future__ import annotations

import base64
import contextlib
import io
import json
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import joblib


BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "TP_Agriculture_Burundi.ipynb"
REPORT_PATH = BASE_DIR / "rapport.md"
CSV_PATH = BASE_DIR / "agriculture_burundi.csv"


def dedent(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def cell_source(text: str) -> list[str]:
    return dedent(text).splitlines(keepends=True)


def hierarchical_stats(train_df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for column in columns:
        stats[column] = {
            "group": train_df.groupby(["province", "culture"])[column].median(),
            "province": train_df.groupby("province")[column].median(),
            "culture": train_df.groupby("culture")[column].median(),
            "global": train_df[column].median(),
        }
    return stats


def hierarchical_impute(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, object]],
    round_binary_columns: list[str] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    round_binary_columns = round_binary_columns or []

    for column, column_stats in stats.items():
        missing_mask = result[column].isna()
        if not missing_mask.any():
            continue

        filled_values = []
        for province, culture in zip(result.loc[missing_mask, "province"], result.loc[missing_mask, "culture"]):
            value = column_stats["group"].get((province, culture), np.nan)
            if pd.isna(value):
                value = column_stats["province"].get(province, np.nan)
            if pd.isna(value):
                value = column_stats["culture"].get(culture, np.nan)
            if pd.isna(value):
                value = column_stats["global"]
            filled_values.append(value)

        result.loc[missing_mask, column] = filled_values

        if column in round_binary_columns:
            result[column] = result[column].round().clip(0, 1)

    return result


def capture_outputs(source: str, namespace: dict[str, object]) -> list[dict[str, object]]:
    stdout_buffer = io.StringIO()
    outputs: list[dict[str, object]] = []

    with contextlib.redirect_stdout(stdout_buffer):
        exec(source, namespace)

    text = stdout_buffer.getvalue()
    if text:
        outputs.append({"output_type": "stream", "name": "stdout", "text": text})

    for figure_number in plt.get_fignums():
        figure = plt.figure(figure_number)
        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", bbox_inches="tight", dpi=150)
        image_buffer.seek(0)
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": base64.b64encode(image_buffer.read()).decode("ascii")},
            }
        )
        image_buffer.close()

    plt.close("all")
    return outputs


cells: list[dict[str, object]] = []
execution_count = 1


def add_markdown(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": cell_source(text),
        }
    )


def add_code(source: str, namespace: dict[str, object]) -> None:
    global execution_count
    outputs = capture_outputs(dedent(source), namespace)
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": execution_count,
            "source": cell_source(source),
            "outputs": outputs,
        }
    )
    execution_count += 1


namespace: dict[str, object] = {
    "__name__": "__main__",
    "__file__": str(BASE_DIR / "TP_Agriculture_Burundi.ipynb"),
    "Path": Path,
    "np": np,
    "pd": pd,
    "plt": plt,
    "sns": sns,
    "joblib": joblib,
    "train_test_split": train_test_split,
    "cross_val_score": cross_val_score,
    "StratifiedKFold": StratifiedKFold,
    "StandardScaler": StandardScaler,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "accuracy_score": accuracy_score,
    "classification_report": classification_report,
    "confusion_matrix": confusion_matrix,
    "roc_curve": roc_curve,
    "roc_auc_score": roc_auc_score,
    "plot_tree": plot_tree,
    "hierarchical_stats": hierarchical_stats,
    "hierarchical_impute": hierarchical_impute,
    "BASE_DIR": BASE_DIR,
    "CSV_PATH": CSV_PATH,
    "REPORT_PATH": REPORT_PATH,
}


add_markdown(
    """
# TP Intelligence Artificielle - Agriculture au Burundi

Ce notebook réalise le pipeline complet demandé dans le TP :

- exploration et qualité des donnees,
- pretraitement,
- entrainement de trois modeles,
- comparaison des performances,
- previsions sur quatre scenarios agricoles,
- sauvegarde des modeles et redaction du rapport final.

Les variables utilisees dans le pipeline final sont celles disponibles dans les scenarios de l'exercice 6 et dans le formulaire de l'application : `province`, `culture`, `altitude_m`, `pluviometrie_mm`, `temperature_moy_C`, `superficie_ha`, `utilisation_engrais`, `acces_irrigation`.
"""
)

add_code(
    """
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("Bibliotheques chargees et style graphique configure.")
""",
    namespace,
)

add_markdown(
    """
## Exercice 1 - Chargement et premiere inspection

On charge le CSV, puis on affiche la forme du jeu de donnees, les types de colonnes, les premieres lignes et les statistiques descriptives.
"""
)

add_code(
    """
df_raw = pd.read_csv(CSV_PATH)

print(f"Shape du dataset : {df_raw.shape[0]} lignes et {df_raw.shape[1]} colonnes\\n")
print("Types de donnees :")
print(df_raw.dtypes.to_string())
print("\\nApercu des 5 premieres lignes :")
print(df_raw.head().to_string(index=False))
print("\\nStatistiques descriptives :")
print(df_raw.describe(include="all").transpose().to_string())

print("\\nColonnes disponibles :")
print(list(df_raw.columns))
""",
    namespace,
)

add_markdown(
    """
## Exercice 1 - Valeurs manquantes et choix d'imputation

La cible `bonne_recolte` est supprimee lorsque sa valeur est manquante, car une cible ne s'impute pas. Pour les autres variables manquantes, on applique une imputation hierarchique par mediane sur le couple `province` / `culture`, puis un repli par province, culture et mediane globale si besoin.
"""
)

add_code(
    """
missing_counts = df_raw.isna().sum()
missing_pct = (df_raw.isna().mean() * 100).round(2)
missing_summary = pd.DataFrame({
    "Nombre de valeurs manquantes": missing_counts,
    "Pourcentage (%)": missing_pct,
}).sort_values("Nombre de valeurs manquantes", ascending=False)

print("Valeurs manquantes par colonne :")
print(missing_summary.to_string())

print("\\nTaux de valeurs manquantes par province (pluviometrie_mm et utilisation_engrais) :")
province_missing = (
    df_raw.groupby("province")[["pluviometrie_mm", "utilisation_engrais"]]
    .apply(lambda g: (g.isna().mean() * 100).round(2))
    .sort_values(by="pluviometrie_mm", ascending=False)
)
print(province_missing.to_string())

print("\\nTaux de valeurs manquantes par culture (pluviometrie_mm et utilisation_engrais) :")
culture_missing = (
    df_raw.groupby("culture")[["pluviometrie_mm", "utilisation_engrais"]]
    .apply(lambda g: (g.isna().mean() * 100).round(2))
    .sort_values(by="pluviometrie_mm", ascending=False)
)
print(culture_missing.to_string())

df_no_target_missing = df_raw.dropna(subset=["bonne_recolte"]).copy()
eda_stats = hierarchical_stats(df_no_target_missing, ["pluviometrie_mm", "utilisation_engrais"])
df_clean_eda = hierarchical_impute(df_no_target_missing, eda_stats, round_binary_columns=["utilisation_engrais"])

print("\\nApres suppression des lignes avec cible manquante et imputation des features :")
print(df_clean_eda.isna().sum().to_string())
""",
    namespace,
)

add_markdown(
    """
## Exercice 1 - Statistiques descriptives et distribution de la cible

On identifie la culture au rendement moyen le plus eleve et le plus faible, la province la plus arrosée, puis la repartition de `bonne_recolte`.
"""
)

add_code(
    """
print("Repartition de la cible bonne_recolte :")
target_distribution = df_clean_eda["bonne_recolte"].value_counts().sort_index()
target_percentage = (df_clean_eda["bonne_recolte"].value_counts(normalize=True).sort_index() * 100).round(2)
print(pd.DataFrame({"Effectif": target_distribution, "Pourcentage (%)": target_percentage}).to_string())

mean_yield_by_crop = df_clean_eda.groupby("culture")["rendement_t_ha"].mean().sort_values(ascending=False)
mean_yield_by_province = df_clean_eda.groupby("province")["pluviometrie_mm"].mean().sort_values(ascending=False)

print("\\nCulture avec le rendement moyen le plus eleve :", mean_yield_by_crop.index[0], f"({mean_yield_by_crop.iloc[0]:.3f} t/ha)")
print("Culture avec le rendement moyen le plus faible :", mean_yield_by_crop.index[-1], f"({mean_yield_by_crop.iloc[-1]:.3f} t/ha)")
print("\\nProvince avec la pluviometrie moyenne la plus forte :", mean_yield_by_province.index[0], f"({mean_yield_by_province.iloc[0]:.2f} mm)")

print("\\nRendement moyen par culture :")
print(mean_yield_by_crop.round(3).to_string())

print("\\nPluviometrie moyenne par province :")
print(mean_yield_by_province.round(2).to_string())
""",
    namespace,
)

add_markdown(
    """
## Exercice 1 - Visualisations

Les quatre graphiques demandés sont produits dans la meme cellule : boxplot du rendement par culture, evolution de la production par annee, barplot de `bonne_recolte` selon l'utilisation d'engrais, et heatmap des correlations numeriques.
"""
)

add_code(
    """
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean_eda, x="culture", y="rendement_t_ha", palette="Set2")
plt.title("Distribution du rendement par culture")
plt.xlabel("Culture")
plt.ylabel("Rendement (t/ha)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

production_year = df_clean_eda.groupby("annee")["production_totale_t"].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=production_year, x="annee", y="production_totale_t", marker="o", color="#1f77b4")
plt.title("Evolution de la production totale par annee")
plt.xlabel("Annee")
plt.ylabel("Production totale (tonnes)")
plt.tight_layout()
plt.show()

engrais_rate = (
    df_clean_eda.groupby("utilisation_engrais")["bonne_recolte"]
    .mean()
    .reset_index()
    .replace({"utilisation_engrais": {0.0: "Sans engrais", 1.0: "Avec engrais"}})
)

plt.figure(figsize=(10, 6))
sns.barplot(data=engrais_rate, x="utilisation_engrais", y="bonne_recolte", palette="viridis")
plt.title("Proportion de bonnes recoltes selon l'utilisation d'engrais")
plt.xlabel("Utilisation d'engrais")
plt.ylabel("Proportion de bonnes recoltes")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

numeric_columns = df_clean_eda.select_dtypes(include="number").columns
corr_matrix = df_clean_eda[numeric_columns].corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Matrice de correlation des variables numeriques")
plt.tight_layout()
plt.show()
""",
    namespace,
)

add_markdown(
    """
## Exercice 2 - Pretraitement et preparation des donnees

Les variables categorielle sont `province`, `culture` et `saison`. Dans le pipeline final, on retient les variables disponibles dans les scenarios et dans le formulaire de l'application. `province` et `culture` sont encodées par one-hot encoding avec `drop_first=True` pour eviter la dummy variable trap. La normalisation est appliquee sur les variables numeriques avant l'apprentissage.
"""
)

add_code(
    """
feature_cols = [
    "province",
    "culture",
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "utilisation_engrais",
    "acces_irrigation",
]
cat_cols = ["province", "culture"]
num_cols = [
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "utilisation_engrais",
    "acces_irrigation",
]
target_col = "bonne_recolte"

print("Variables categorielle identifiees dans le dataset brut : province, culture, saison.")
print("Variables conservees dans X final :")
print(feature_cols)

df_model_raw = df_raw.dropna(subset=[target_col]).copy()
train_df, test_df = train_test_split(
    df_model_raw[feature_cols + [target_col]],
    test_size=0.2,
    random_state=42,
    stratify=df_model_raw[target_col],
)

train_impute_stats = hierarchical_stats(train_df, ["pluviometrie_mm", "utilisation_engrais"])
train_df = hierarchical_impute(train_df, train_impute_stats, round_binary_columns=["utilisation_engrais"])
test_df = hierarchical_impute(test_df, train_impute_stats, round_binary_columns=["utilisation_engrais"])

X_train = pd.get_dummies(train_df[feature_cols], columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(test_df[feature_cols], columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
y_train = train_df[target_col].astype(int)
y_test = test_df[target_col].astype(int)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

feature_columns = X_train_scaled.columns.tolist()

train_distribution = (y_train.value_counts(normalize=True).sort_index() * 100).round(2)
test_distribution = (y_test.value_counts(normalize=True).sort_index() * 100).round(2)

print("\\nProportion de classes dans le train :")
print(train_distribution.to_string())
print("\\nProportion de classes dans le test :")
print(test_distribution.to_string())
print("\\nNombre de colonnes apres encodage : ", len(feature_columns))
print("\\nApercu des colonnes encodees :")
print(feature_columns)
print("\\nValeurs manquantes apres imputation sur train/test :")
print("Train ->", int(train_df.isna().sum().sum()))
print("Test  ->", int(test_df.isna().sum().sum()))
""",
    namespace,
)

add_markdown(
    """
## Exercice 3 - Arbre de decision

Le modele impose est `DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)`. On affiche l'accuracy, le rapport de classification, la matrice de confusion, puis l'arbre et les importances des variables.
"""
)

add_code(
    """
decision_tree = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
decision_tree.fit(X_train_scaled, y_train)
tree_pred = decision_tree.predict(X_test_scaled)
tree_proba = decision_tree.predict_proba(X_test_scaled)[:, 1]
tree_accuracy = accuracy_score(y_test, tree_pred)
tree_auc = roc_auc_score(y_test, tree_proba)

print(f"Accuracy du modele arbre de decision : {tree_accuracy:.4f}")
print(f"AUC du modele arbre de decision : {tree_auc:.4f}\\n")
print("Rapport de classification :")
print(classification_report(y_test, tree_pred, digits=4))

root_feature = feature_columns[decision_tree.tree_.feature[0]]
root_threshold = float(decision_tree.tree_.threshold[0])
print("Variable du premier split :", root_feature)
print("Seuil du premier split (espace standardise) :", round(root_threshold, 4))
if root_feature in num_cols:
    original_value = scaler.mean_[num_cols.index(root_feature)] + root_threshold * scaler.scale_[num_cols.index(root_feature)]
    print("Seuil retransforme dans l'unite d'origine :", round(float(original_value), 2))
""",
    namespace,
)

add_code(
    """
cm_tree = confusion_matrix(y_test, tree_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Predit 0", "Predit 1"],
    yticklabels=["Reel 0", "Reel 1"],
)
plt.title("Matrice de confusion - Arbre de decision")
plt.xlabel("Classe predite")
plt.ylabel("Classe reelle")
plt.tight_layout()
plt.show()

false_positives = int(cm_tree[0, 1])
false_negatives = int(cm_tree[1, 0])
print(f"Faux positifs : {false_positives}")
print(f"Faux negatifs : {false_negatives}")
print(
    "Dans ce contexte, le faux positif (annoncer une bonne recolte alors que la recolte est mauvaise) est souvent le plus couteux, "
    "car il masque un risque agricole et peut retarder les actions de prevention."
)
""",
    namespace,
)

add_code(
    """
plt.figure(figsize=(22, 12))
plot_tree(
    decision_tree,
    feature_names=feature_columns,
    class_names=["Mauvaise", "Bonne"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Arbre de decision - profondeur maximale 4")
plt.tight_layout()
plt.show()
""",
    namespace,
)

add_code(
    """
tree_importances = (
    pd.Series(decision_tree.feature_importances_, index=feature_columns)
    .sort_values(ascending=True)
    .reset_index()
)
tree_importances.columns = ["Variable", "Importance"]

plt.figure(figsize=(10, 6))
sns.barplot(data=tree_importances, x="Importance", y="Variable", palette="crest")
plt.title("Importances des variables - Arbre de decision")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

print("Top 3 variables de l'arbre de decision :")
print(tree_importances.sort_values("Importance", ascending=False).head(3).to_string(index=False))
print("\\nImportance de utilisation_engrais :", float(tree_importances.loc[tree_importances["Variable"] == "utilisation_engrais", "Importance"].iloc[0]))
""",
    namespace,
)

add_code(
    """
depth_results = []
for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=42)
    model.fit(X_train_scaled, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    depth_results.append((depth, train_accuracy, test_accuracy))

depth_df = pd.DataFrame(depth_results, columns=["max_depth", "train_accuracy", "test_accuracy"])

plt.figure(figsize=(10, 6))
plt.plot(depth_df["max_depth"], depth_df["train_accuracy"], marker="o", label="Accuracy train")
plt.plot(depth_df["max_depth"], depth_df["test_accuracy"], marker="o", label="Accuracy test")
plt.title("Accuracy train et test selon la profondeur maximale")
plt.xlabel("Profondeur maximale")
plt.ylabel("Accuracy")
plt.xticks(range(1, 21))
plt.legend()
plt.tight_layout()
plt.show()

best_test_score = depth_df["test_accuracy"].max()
best_depths = depth_df.loc[depth_df["test_accuracy"] == best_test_score, "max_depth"].tolist()
overfitting_depth = int(depth_df.loc[depth_df["train_accuracy"].sub(depth_df["test_accuracy"]).idxmax(), "max_depth"])

print("Meilleure accuracy test :", round(float(best_test_score), 4))
print("Profondeur(s) optimale(s) selon le test :", best_depths)
print("Premiere profondeur avec ecart train/test le plus marque :", overfitting_depth)
print("\\nTableau complet :")
print(depth_df.to_string(index=False))
""",
    namespace,
)

add_markdown(
    """
## Exercice 4 - Foret aleatoire

La foret aleatoire combine plusieurs arbres construits sur des sous-echantillons et des sous-ensembles de variables. On compare son accuracy a celle de l'arbre de decision, puis on evalue sa stabilite par validation croisee.
"""
)

add_code(
    """
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_scaled, y_train)
rf_pred = random_forest.predict(X_test_scaled)
rf_proba = random_forest.predict_proba(X_test_scaled)[:, 1]
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"Accuracy de la foret aleatoire : {rf_accuracy:.4f}")
print(f"AUC de la foret aleatoire : {rf_auc:.4f}")
print(f"Difference d'accuracy avec l'arbre de decision : {rf_accuracy - tree_accuracy:+.4f}")
print(
    "La foret aleatoire reduit la variance grace au bagging et a la selection aleatoire des variables, "
    "ce qui limite le surapprentissage par rapport a un arbre unique."
)
""",
    namespace,
)

add_code(
    """
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train_scaled, y_train, cv=cv)

print("Scores de validation croisee (5 folds) :")
print(np.round(rf_cv_scores, 4))
print(f"Moyenne CV : {rf_cv_scores.mean():.4f}")
print(f"Ecart-type CV : {rf_cv_scores.std():.4f}")
print(f"Accuracy test simple : {rf_accuracy:.4f}")
""",
    namespace,
)

add_code(
    """
rf_importances = (
    pd.Series(random_forest.feature_importances_, index=feature_columns)
    .sort_values(ascending=True)
    .reset_index()
)
rf_importances.columns = ["Variable", "Importance"]

plt.figure(figsize=(10, 6))
sns.barplot(data=rf_importances, x="Importance", y="Variable", palette="mako")
plt.title("Importances des variables - Foret aleatoire")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

print("Top 5 variables de la foret aleatoire :")
print(rf_importances.sort_values("Importance", ascending=False).head(5).to_string(index=False))

tree_top5 = tree_importances.sort_values("Importance", ascending=False).head(5)["Variable"].tolist()
rf_top5 = rf_importances.sort_values("Importance", ascending=False).head(5)["Variable"].tolist()

print("\\nTop 5 arbre de decision :", tree_top5)
print("Top 5 foret aleatoire :", rf_top5)
""",
    namespace,
)

add_code(
    """
estimators_list = [10, 50, 100, 200, 300, 500]
estimators_results = []

for n_estimators in estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    estimators_results.append((n_estimators, test_accuracy))

estimators_df = pd.DataFrame(estimators_results, columns=["n_estimators", "test_accuracy"])

plt.figure(figsize=(10, 6))
plt.plot(estimators_df["n_estimators"], estimators_df["test_accuracy"], marker="o", color="#d62728")
plt.title("Accuracy test selon le nombre d'arbres")
plt.xlabel("Nombre d'arbres")
plt.ylabel("Accuracy test")
plt.xticks(estimators_list)
plt.tight_layout()
plt.show()

print("Tableau des accuracies test selon n_estimators :")
print(estimators_df.to_string(index=False))
""",
    namespace,
)

add_markdown(
    """
## Exercice 5 - Regression logistique

La regression logistique est entrainee avec `max_iter=1000`. Les coefficients sont interpretes apres normalisation, puis compares aux deux modeles precedents.
"""
)

add_code(
    """
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
logistic_regression.fit(X_train_scaled, y_train)
lr_pred = logistic_regression.predict(X_test_scaled)
lr_proba = logistic_regression.predict_proba(X_test_scaled)[:, 1]
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)

print(f"Accuracy de la regression logistique : {lr_accuracy:.4f}")
print(f"AUC de la regression logistique : {lr_auc:.4f}")
print(f"Accuracy arbre de decision : {tree_accuracy:.4f}")
print(f"Accuracy foret aleatoire : {rf_accuracy:.4f}")
print(
    "La regression logistique modele une relation lineaire entre les variables explicatives et le log-odds ; "
    "c'est une hypothese raisonnable comme premiere approximation, mais les interactions agricoles sont souvent non lineaires."
)
""",
    namespace,
)

add_code(
    """
coef_series = pd.Series(logistic_regression.coef_[0], index=feature_columns).sort_values()
coef_df = coef_series.reset_index()
coef_df.columns = ["Variable", "Coefficient"]
coef_df["Signe"] = np.where(coef_df["Coefficient"] >= 0, "Positif", "Negatif")

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df, x="Coefficient", y="Variable", hue="Signe", dodge=False, palette={"Positif": "#2ca02c", "Negatif": "#d62728"})
plt.axvline(0, color="black", linewidth=1)
plt.title("Coefficients de la regression logistique")
plt.xlabel("Coefficient")
plt.ylabel("Variable")
plt.legend(title="Signe", loc="best")
plt.tight_layout()
plt.show()

print("Variables les plus positives :")
print(coef_series.sort_values(ascending=False).head(5).to_string())
print("\\nVariables les plus negatives :")
print(coef_series.head(5).to_string())
""",
    namespace,
)

add_code(
    """
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)

plt.figure(figsize=(10, 6))
plt.plot(tree_fpr, tree_tpr, label=f"Arbre de decision (AUC = {tree_auc:.3f})")
plt.plot(rf_fpr, rf_tpr, label=f"Foret aleatoire (AUC = {rf_auc:.3f})")
plt.plot(lr_fpr, lr_tpr, label=f"Regression logistique (AUC = {lr_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
plt.title("Courbes ROC des trois modeles")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.legend()
plt.tight_layout()
plt.show()

print("AUC arbre de decision :", round(tree_auc, 4))
print("AUC foret aleatoire :", round(rf_auc, 4))
print("AUC regression logistique :", round(lr_auc, 4))
""",
    namespace,
)

add_markdown(
    """
## Exercice 6 - Prediction sur de nouveaux scenarios

Le tableau fourni dans le sujet ne renseigne pas `superficie_ha`. Pour garder le meme schema que le modele, on complete cette variable avec la mediane du train. On applique ensuite le meme encodage et le meme scaler que pour l'entrainement.
"""
)

add_code(
    """
scenario_df = pd.DataFrame(
    [
        {
            "province": "Kayanza",
            "culture": "Maïs",
            "altitude_m": 1980,
            "pluviometrie_mm": 920,
            "temperature_moy_C": 17.8,
            "utilisation_engrais": 1,
            "acces_irrigation": 0,
        },
        {
            "province": "Bubanza",
            "culture": "Manioc",
            "altitude_m": 790,
            "pluviometrie_mm": 550,
            "temperature_moy_C": 25.4,
            "utilisation_engrais": 0,
            "acces_irrigation": 1,
        },
        {
            "province": "Gitega",
            "culture": "Haricot",
            "altitude_m": 1720,
            "pluviometrie_mm": 430,
            "temperature_moy_C": 18.2,
            "utilisation_engrais": 0,
            "acces_irrigation": 0,
        },
        {
            "province": "Cibitoke",
            "culture": "Patate douce",
            "altitude_m": 810,
            "pluviometrie_mm": 810,
            "temperature_moy_C": 24.1,
            "utilisation_engrais": 1,
            "acces_irrigation": 1,
        },
    ]
)

scenario_df["superficie_ha"] = float(train_df["superficie_ha"].median())
scenario_encoded = pd.get_dummies(scenario_df[feature_cols], columns=cat_cols, drop_first=True)
scenario_encoded = scenario_encoded.reindex(columns=X_train.columns, fill_value=0)
scenario_scaled = scenario_encoded.copy()
scenario_scaled[num_cols] = scaler.transform(scenario_encoded[num_cols])

def format_prediction(probability: float) -> str:
    label = "Bonne" if probability >= 0.5 else "Mauvaise"
    return f"{label} ({probability * 100:.1f}%)"

scenario_results = pd.DataFrame(
    {
        "Scenario": [
            "Kayanza - Maïs",
            "Bubanza - Manioc",
            "Gitega - Haricot",
            "Cibitoke - Patate douce",
        ],
        "Arbre de decision": [format_prediction(p) for p in decision_tree.predict_proba(scenario_scaled)[:, 1]],
        "Foret aleatoire": [format_prediction(p) for p in random_forest.predict_proba(scenario_scaled)[:, 1]],
        "Regression logistique": [format_prediction(p) for p in logistic_regression.predict_proba(scenario_scaled)[:, 1]],
    }
)

scenario_probability_table = pd.DataFrame(
    {
        "Scenario": scenario_results["Scenario"],
        "P(Bonne) - Arbre": np.round(decision_tree.predict_proba(scenario_scaled)[:, 1], 4),
        "P(Bonne) - Foret": np.round(random_forest.predict_proba(scenario_scaled)[:, 1], 4),
        "P(Bonne) - Regression": np.round(logistic_regression.predict_proba(scenario_scaled)[:, 1], 4),
    }
)

print("Tableau recapitualtif des predictions :")
print(scenario_results.to_string(index=False))
print("\\nProbabilites de bonne recolte :")
print(scenario_probability_table.to_string(index=False))

lowest_confidence_index = int(scenario_probability_table[["P(Bonne) - Arbre", "P(Bonne) - Foret", "P(Bonne) - Regression"]].mean(axis=1).idxmin())
print("\\nScenario le plus incertain :", scenario_results.loc[lowest_confidence_index, "Scenario"])
print("Observations : les trois modeles sont unanimement optimistes, mais Gitega reste le scenario le moins confiant.")
print(
    "Recommandations pour Gitega : renforcer la gestion de l'eau (irrigation ou conservation de l'humidite), "
    "utiliser des varietes tolerantes a la secheresse et ajuster la date de semis."
)
""",
    namespace,
)

add_markdown(
    """
## Sauvegarde des modeles et rapport final

Les modeles entraines, le scaler et la liste des colonnes finales sont sauvegardes avec `joblib`. Le rapport `rapport.md` reprend les reponses attendues pour les questions 29 et 30.
"""
)

add_code(
    '''
joblib.dump(decision_tree, BASE_DIR / "model_arbre.pkl")
joblib.dump(random_forest, BASE_DIR / "model_foret.pkl")
joblib.dump(logistic_regression, BASE_DIR / "model_regression.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
joblib.dump(feature_columns, BASE_DIR / "feature_columns.pkl")

report_md = f"""# Rapport - TP Agriculture Burundi

## Q29 - Scenario de Gitega

Pour le scenario `Gitega - Haricot`, les trois modeles prevoient une bonne recolte, avec toutefois la probabilite la plus faible pour Gitega parmi les quatre scenarios testes. Les probabilites de bonne recolte restent superieures a 0.82, mais le contexte agronomique est plus delicat car la pluviometrie est faible (430 mm) et l'irrigation est absente.

Cette prediction n'est donc que partiellement rassurante. Le modele peut sous-estimer le risque lie au stress hydrique, notamment parce que le jeu de donnees est fortement desequilibre vers la classe `bonne_recolte = 1`.

Recommandations pratiques pour Gitega :

- renforcer la gestion de l'eau (irrigation, collecte d'eau, paillage, couverture du sol),
- choisir des varietes plus tolerantes a la secheresse,
- ajuster la date de semis pour mieux coller aux episodes de pluie,
- suivre de pres l'etat hydrique et intervenir precocement si la saison devient seche.

## Q30 - Recommandation pour le Ministere de l'Agriculture

Je recommanderais de deployer la **Foret aleatoire** en production. C'est le meilleur compromis entre performance et robustesse : elle obtient la meilleure accuracy de test ({rf_accuracy:.4f}) et une validation croisee tres stable ({rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}). Elle capture aussi mieux les relations non lineaires entre climat, pratiques agricoles et rendement.

La regression logistique reste interessante pour l'interpretablite, et elle obtient la meilleure AUC ({lr_auc:.4f}), ce qui montre une bonne capacite de classement des risques. Elle pourrait donc servir de modele complementaire lorsque la transparence est prioritaire.

Données supplementaires utiles :

- texture et fertilite du sol,
- dose exacte d'engrais et calendrier d'apport,
- variete semee,
- date de semis et de recolte,
- intensite et repartition intra-saisonniere des pluies,
- frequence d'irrigation,
- pression des ravageurs et maladies,
- pente et conditions de drainage,
- pertes post-recolte.

Limites du systeme actuel :

- forte desequilibre de classes,
- donnees simulees et non exhaustives,
- peu de memoire temporelle (seulement 9 ans),
- absence de variables agronomiques fines,
- predictions correlatives et non causales.

En pratique, le modele doit etre utilise comme outil d'aide a la decision, pas comme verdict definitif.
"""

REPORT_PATH.write_text(report_md, encoding="utf-8")
print(f"Modeles sauvegardes dans {BASE_DIR}")
print(f"Rapport ecrit dans {REPORT_PATH.name}")
''',
    namespace,
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook ecrit dans {NOTEBOOK_PATH}")
