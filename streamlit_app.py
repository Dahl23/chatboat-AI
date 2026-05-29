from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "agriculture_burundi.csv"
ARBRE_PATH = BASE_DIR / "model_arbre.pkl"
FORET_PATH = BASE_DIR / "model_foret.pkl"
REGRESSION_PATH = BASE_DIR / "model_regression.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

TARGET_COL = "bonne_recolte"
CAT_COLS = ["province", "culture"]
NUM_COLS = [
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "utilisation_engrais",
    "acces_irrigation",
]
FEATURE_INPUT_COLS = [
    "province",
    "culture",
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "utilisation_engrais",
    "acces_irrigation",
]

MODEL_LABELS = {
    "Arbre de décision": "Arbre de décision",
    "Forêt aléatoire": "Forêt aléatoire",
    "Régression logistique": "Régression logistique",
}

MODEL_NOTES = {
    "Arbre de décision": "Modèle très lisible, mais plus sensible au surapprentissage qu'une forêt aléatoire.",
    "Forêt aléatoire": "Meilleur compromis global entre précision et robustesse sur ce TP.",
    "Régression logistique": "Modèle simple et interprétable, avec la meilleure AUC sur le jeu de test.",
}

CULTURE_DISPLAY_TO_INTERNAL = {
    "Maïs": "Maļs",
    "Haricot": "Haricot",
    "Manioc": "Manioc",
    "Patate douce": "Patate douce",
    "Sorgho": "Sorgho",
    "Bananier": "Bananier",
}

CULTURE_INTERNAL_TO_DISPLAY = {value: key for key, value in CULTURE_DISPLAY_TO_INTERNAL.items()}

NUMERICAL_LABELS = {
    "altitude_m": "Altitude (m)",
    "pluviometrie_mm": "Pluviométrie (mm)",
    "temperature_moy_C": "Température moyenne (°C)",
    "superficie_ha": "Superficie (ha)",
    "utilisation_engrais": "Utilisation d'engrais",
    "acces_irrigation": "Accès à l'irrigation",
}

st.set_page_config(
    page_title="TP Agriculture Burundi",
    page_icon="🌾",
    layout="wide",
)

sns.set_theme(style="whitegrid", context="notebook")


def pretty_feature_name(name: str) -> str:
    return (
        name.replace("province_", "Province : ")
        .replace("culture_", "Culture : ")
        .replace("Maļs", "Maïs")
    )


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

        values = []
        for province, culture in zip(result.loc[missing_mask, "province"], result.loc[missing_mask, "culture"]):
            value = column_stats["group"].get((province, culture), np.nan)
            if pd.isna(value):
                value = column_stats["province"].get(province, np.nan)
            if pd.isna(value):
                value = column_stats["culture"].get(culture, np.nan)
            if pd.isna(value):
                value = column_stats["global"]
            values.append(value)

        result.loc[missing_mask, column] = values

        if column in round_binary_columns:
            result[column] = result[column].round().clip(0, 1)

    return result


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[dict[str, object], object, list[str]]:
    models = {
        "Arbre de décision": joblib.load(ARBRE_PATH),
        "Forêt aléatoire": joblib.load(FORET_PATH),
        "Régression logistique": joblib.load(REGRESSION_PATH),
    }
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return models, scaler, feature_columns


def encode_frame(frame: pd.DataFrame, feature_columns: list[str], scaler: object) -> pd.DataFrame:
    encoded = pd.get_dummies(frame[FEATURE_INPUT_COLS], columns=CAT_COLS, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = encoded.copy()
    scaled[NUM_COLS] = scaler.transform(encoded[NUM_COLS])
    return scaled


@st.cache_data(show_spinner=False)
def build_reference_split(feature_columns: tuple[str, ...]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = load_dataset().dropna(subset=[TARGET_COL]).copy()
    train_df, test_df = train_test_split(
        df[FEATURE_INPUT_COLS + [TARGET_COL]],
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL],
    )

    train_stats = hierarchical_stats(train_df, ["pluviometrie_mm", "utilisation_engrais"])
    train_df = hierarchical_impute(train_df, train_stats, round_binary_columns=["utilisation_engrais"])
    test_df = hierarchical_impute(test_df, train_stats, round_binary_columns=["utilisation_engrais"])

    _, scaler, _ = load_artifacts()
    X_train = encode_frame(train_df, list(feature_columns), scaler)
    X_test = encode_frame(test_df, list(feature_columns), scaler)
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)
    return X_train, y_train, X_test, y_test


def compute_model_metrics(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}

    for model_name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metric_rows.append(
            {
                "Modèle": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "F1": f1_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, probabilities),
            }
        )
        details[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "report": classification_report(y_test, predictions, output_dict=True),
        }

    metrics_df = pd.DataFrame(metric_rows).set_index("Modèle").sort_values("Accuracy", ascending=False)
    return metrics_df, details


def build_feature_chart(model_name: str, model: object, feature_columns: list[str]) -> tuple[plt.Figure, pd.DataFrame]:
    if hasattr(model, "coef_"):
        values = pd.Series(model.coef_[0], index=feature_columns).sort_values()
        chart_df = values.reset_index()
        chart_df.columns = ["Variable", "Valeur"]
        chart_df["Signe"] = np.where(chart_df["Valeur"] >= 0, "Positif", "Négatif")
        chart_df["Variable"] = chart_df["Variable"].map(pretty_feature_name)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=chart_df,
            x="Valeur",
            y="Variable",
            hue="Signe",
            dodge=False,
            palette={"Positif": "#2ca02c", "Négatif": "#d62728"},
            ax=ax,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Coefficients de {model_name}")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Variable")
        ax.legend(title="Signe")
        fig.tight_layout()
        return fig, chart_df

    values = pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_columns))), index=feature_columns)
    chart_df = values.sort_values(ascending=True).reset_index()
    chart_df.columns = ["Variable", "Valeur"]
    chart_df["Variable"] = chart_df["Variable"].map(pretty_feature_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=chart_df, x="Valeur", y="Variable", palette="crest", ax=ax)
    ax.set_title(f"Importances des variables - {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    return fig, chart_df


def display_model_information(model_name: str, metrics_df: pd.DataFrame, metrics_detail: dict[str, dict[str, object]]) -> None:
    row = metrics_df.loc[model_name]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{row['Accuracy']:.3f}")
    col2.metric("F1-score", f"{row['F1']:.3f}")
    col3.metric("AUC", f"{row['AUC']:.3f}")

    with st.expander("Voir l'évaluation détaillée du modèle sélectionné", expanded=True):
        detail = metrics_detail[model_name]
        cm = detail["confusion_matrix"]
        report_df = pd.DataFrame(detail["report"]).transpose()
        report_df = report_df.rename(index={"0": "Mauvaise récolte", "1": "Bonne récolte", "0.0": "Mauvaise récolte", "1.0": "Bonne récolte"})
        st.write("Rapport de classification sur le jeu de test")
        st.dataframe(report_df.round(3), use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Prédit 0", "Prédit 1"],
            yticklabels=["Réel 0", "Réel 1"],
            ax=ax,
        )
        ax.set_title("Matrice de confusion")
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


def format_prediction(probability: float) -> str:
    return "Bonne récolte" if probability >= 0.5 else "Mauvaise récolte"


def build_single_prediction(
    province: str,
    culture_display: str,
    altitude_m: float,
    pluviometrie_mm: float,
    temperature_moy_C: float,
    superficie_ha: float,
    utilisation_engrais: int,
    acces_irrigation: int,
    scaler: object,
    feature_columns: list[str],
) -> pd.DataFrame:
    sample = pd.DataFrame(
        [
            {
                "province": province,
                "culture": CULTURE_DISPLAY_TO_INTERNAL[culture_display],
                "altitude_m": altitude_m,
                "pluviometrie_mm": pluviometrie_mm,
                "temperature_moy_C": temperature_moy_C,
                "superficie_ha": superficie_ha,
                "utilisation_engrais": utilisation_engrais,
                "acces_irrigation": acces_irrigation,
            }
        ]
    )
    return encode_frame(sample, feature_columns, scaler)


def main() -> None:
    models, scaler, feature_columns = load_artifacts()
    feature_columns = list(feature_columns)
    X_train, y_train, X_test, y_test = build_reference_split(tuple(feature_columns))
    metrics_df, metrics_detail = compute_model_metrics(models, X_test, y_test)

    dataset = load_dataset()
    provinces = sorted(dataset["province"].dropna().unique().tolist())
    cultures_display = list(CULTURE_DISPLAY_TO_INTERNAL.keys())

    st.title("Prédiction des récoltes au Burundi")
    st.caption(
        "Application web du TP d'IA appliquée à l'agriculture. "
        "Le modèle choisi prédit si la récolte sera bonne ou mauvaise à partir des caractéristiques de la parcelle."
    )

    st.sidebar.header("Paramètres")
    selected_model = st.sidebar.selectbox("Choisir le modèle", list(MODEL_LABELS.keys()))
    st.sidebar.info(MODEL_NOTES[selected_model])

    st.sidebar.subheader("Aperçu du jeu de données")
    sidebar_cols = st.sidebar.columns(2)
    sidebar_cols[0].metric("Lignes", f"{len(dataset):,}".replace(",", " "))
    sidebar_cols[1].metric("Colonnes", f"{dataset.shape[1]}")
    st.sidebar.metric("Provinces", dataset["province"].nunique())
    st.sidebar.metric("Cultures", dataset["culture"].nunique())
    st.sidebar.metric("Classe bonne récolte", f"{dataset[TARGET_COL].dropna().mean():.1%}")

    st.sidebar.subheader("Comparaison des modèles")
    st.sidebar.dataframe(metrics_df.round(3), use_container_width=True)

    st.subheader("1. Saisie de la parcelle")
    st.write(
        "Renseignez les champs ci-dessous puis cliquez sur **Prédire**. "
        "Les variables utilisées sont celles du pipeline d'entraînement."
    )

    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            province = st.selectbox("Province", provinces)
            culture_display = st.selectbox("Culture", cultures_display)
            altitude_m = st.number_input("Altitude (m)", min_value=500.0, max_value=2500.0, value=1750.0, step=10.0)
            pluviometrie_mm = st.number_input("Pluviométrie (mm)", min_value=0.0, max_value=2000.0, value=800.0, step=10.0)

        with form_col2:
            temperature_moy_C = st.number_input("Température moyenne (°C)", min_value=10.0, max_value=35.0, value=20.0, step=0.1)
            superficie_ha = st.number_input("Superficie (ha)", min_value=0.1, max_value=20.0, value=2.4, step=0.1)
            utilisation_engrais = st.selectbox("Utilisation d'engrais", [0, 1], format_func=lambda value: "Oui" if value == 1 else "Non")
            acces_irrigation = st.selectbox("Accès à l'irrigation", [0, 1], format_func=lambda value: "Oui" if value == 1 else "Non")

        submitted = st.form_submit_button("Prédire")

    st.subheader("2. Modèle sélectionné")
    display_model_information(selected_model, metrics_df, metrics_detail)

    if submitted:
        selected_model_object = models[selected_model]
        sample_scaled = build_single_prediction(
            province=province,
            culture_display=culture_display,
            altitude_m=float(altitude_m),
            pluviometrie_mm=float(pluviometrie_mm),
            temperature_moy_C=float(temperature_moy_C),
            superficie_ha=float(superficie_ha),
            utilisation_engrais=int(utilisation_engrais),
            acces_irrigation=int(acces_irrigation),
            scaler=scaler,
            feature_columns=feature_columns,
        )

        probability = float(selected_model_object.predict_proba(sample_scaled)[0, 1])
        prediction = format_prediction(probability)
        label_colour = "success" if probability >= 0.5 else "error"

        st.subheader("3. Résultat de la prédiction")
        if label_colour == "success":
            st.success(f"Prédiction : {prediction}")
        else:
            st.error(f"Prédiction : {prediction}")

        result_cols = st.columns(3)
        result_cols[0].metric("Probabilité de bonne récolte", f"{probability:.1%}")
        result_cols[1].metric("Probabilité de mauvaise récolte", f"{(1 - probability):.1%}")
        result_cols[2].metric("Seuil de décision", "50%")

        st.progress(min(max(probability, 0.0), 1.0))

        sample_summary = pd.DataFrame(
            [
                {
                    "Province": province,
                    "Culture": culture_display,
                    "Altitude (m)": altitude_m,
                    "Pluviométrie (mm)": pluviometrie_mm,
                    "Température (°C)": temperature_moy_C,
                    "Superficie (ha)": superficie_ha,
                    "Engrais": "Oui" if utilisation_engrais else "Non",
                    "Irrigation": "Oui" if acces_irrigation else "Non",
                }
            ]
        )
        st.dataframe(sample_summary, use_container_width=True)

        st.subheader("4. Variables influentes")
        fig, chart_df = build_feature_chart(selected_model, selected_model_object, feature_columns)
        st.pyplot(fig, clear_figure=True)

        top_variables = chart_df.sort_values("Valeur", ascending=False).head(5)
        st.write("Top 5 des variables selon le modèle sélectionné")
        st.dataframe(top_variables, use_container_width=True)
    else:
        st.info("Remplissez le formulaire puis cliquez sur **Prédire** pour obtenir le résultat.")

    with st.expander("À propos du modèle et de la logique de prédiction"):
        st.markdown(
            """
            - **Arbre de décision** : facile à interpréter, mais plus fragile.
            - **Forêt aléatoire** : agrégation d'arbres pour réduire la variance.
            - **Régression logistique** : modèle linéaire probabiliste, utile pour classer le risque.

            L'app réutilise les modèles sauvegardés (`.pkl`), le scaler et les colonnes finales du TP.
            """
        )


if __name__ == "__main__":
    main()
