# TP Agriculture Burundi - Prédiction des Récoltes au Burundi

Projet complet de Machine Learning appliqué à l'agriculture au Burundi. Ce projet développe des modèles prédictifs pour évaluer si une récolte sera bonne ou mauvaise en fonction de facteurs géographiques et agronomiques.

## Contexte

L'agriculture représente plus de 40% du PIB du Burundi et constitue la principale source de revenus pour la population rurale. Ce projet vise à améliorer les rendements agricoles en utilisant des modèles de classification basés sur des données historiques de 2015-2023.

## Architecture du projet

### Pipeline ML complet
- **Exercice 1** : Chargement, exploration et qualité des données (1 620 observations, 15 provinces, 6 cultures)
- **Exercice 2** : Prétraitement et préparation des données (encodage, normalisation, train/test split)
- **Exercice 3** : Arbre de décision avec analyse de l'overfitting
- **Exercice 4** : Forêt aléatoire avec validation croisée (5-fold)
- **Exercice 5** : Régression logistique avec analyse ROC/AUC
- **Exercice 6** : Prédictions sur 4 scénarios agricoles réels
- **Exercice 7** : Application web Streamlit

### Modèles entraînés

| Modèle | Accuracy | F1-Score | AUC | Avantage |
|--------|----------|----------|-----|----------|
| **Forêt aléatoire** | 0.9367 | Excellent | 0.8155 | Meilleur compromis robustesse/précision (RECOMMANDÉ) |
| Arbre de décision | 0.9245 | Bon | 0.7842 | Haute interprétabilité |
| Régression logistique | 0.8903 | Bon | 0.8458 | Meilleure AUC, très transparente |

## Contenu des fichiers

- `TP_Agriculture_Burundi.ipynb` : notebook Jupyter complet avec toutes les analyses
- `streamlit_app.py` : application web interactive pour prédictions en temps réel
- `agriculture_burundi.csv` : dataset complet (1 620 lignes)
- `model_arbre.pkl`, `model_foret.pkl`, `model_regression.pkl` : modèles entraînés
- `scaler.pkl` : normaliseur (StandardScaler) pour les variables numériques
- `feature_columns.pkl` : liste des colonnes finales après encodage
- `rapport.md` : analyse approfondie et recommandations
- `requirements.txt` : dépendances Python
- `runtime.txt` : spécification Python pour déploiement

## Lancer l'application en local

### Prérequis
- Python 3.8+
- pip ou conda

### Installation et lancement

1. Cloner le dépôt et naviguer dans le répertoire :
```bash
git clone https://github.com/votre-username/chatboat-AI.git
cd chatboat-AI
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancer l'application Streamlit :
```bash
streamlit run streamlit_app.py
```

L'application sera accessible à `http://localhost:8501`

### Utilisation

1. **Saisir les paramètres de la parcelle** :
   - Province (15 provinces du Burundi)
   - Culture (6 cultures principales : Maïs, Haricot, Manioc, Patate douce, Sorgho, Bananier)
   - Altitude (500-2 500 m)
   - Pluviométrie (0-2 000 mm)
   - Température moyenne (10-35 °C)
   - Superficie (0.1-20 ha)
   - Utilisation d'engrais (Oui/Non)
   - Accès à l'irrigation (Oui/Non)

2. **Sélectionner le modèle** (dans la barre latérale)

3. **Obtenir la prédiction** avec probabilité associée

4. **Visualiser l'importance des variables** selon le modèle choisi

## Déploiement sur Streamlit Community Cloud

### Option 1 : Via l'interface web (recommandé)

1. Créer un dépôt GitHub **public** avec ces fichiers à la racine :
   - `streamlit_app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `README.md`
   - `agriculture_burundi.csv`
   - `model_arbre.pkl`
   - `model_foret.pkl`
   - `model_regression.pkl`
   - `scaler.pkl`
   - `feature_columns.pkl`

2. Se connecter à [Streamlit Community Cloud](https://share.streamlit.io/)

3. Cliquer sur **Create app**

4. Sélectionner :
   - Votre dépôt GitHub
   - Branche : `main`
   - Chemin du fichier : `streamlit_app.py`

5. Attendre la fin du build (2-3 minutes)

### Option 2 : Via Git en ligne de commande

```bash
git init -b main
git add .
git commit -m "Initial commit: ML prediction app"
git remote add origin https://github.com/VOTRE USERNAME/chatboat-AI.git
git push -u origin main
```

Ensuite connecter le dépôt à Streamlit Community Cloud via l'interface web.

### Points de vérification

- ✅ `streamlit_app.py` à la racine
- ✅ Tous les fichiers `.pkl` à la racine
- ✅ `agriculture_burundi.csv` présent
- ✅ `requirements.txt` à jour
- ✅ `runtime.txt` specifies Python 3.10+
- ✅ Dépôt GitHub en mode **PUBLIC**

En cas d'erreur au déploiement, consulter les logs Streamlit (lien fourni par l'interface)

## Stack technologique

### Libraries ML
- **scikit-learn** : Modèles (DecisionTree, RandomForest, LogisticRegression)
- **pandas** : Manipulation des données
- **numpy** : Calculs numériques

### Visualisation
- **matplotlib** : Graphiques statiques
- **seaborn** : Visualisations statistiques

### Web
- **Streamlit** : Framework web minimaliste
- **joblib** : Sérialisation des modèles

## Utilisation des modèles

### Modèle recommandé : Forêt aléatoire
- **Performance** : 93.67% accuracy en test
- **Robustesse** : Validation croisée stable (0.9341 ± 0.0032)
- **Capacité** : Capture les relations non-linéaires entre variables
- **Utilisation** : Décisions opérationnelles en production

### Modèle complémentaire : Régression logistique
- **Performance** : AUC maximale (0.8458)
- **Avantage** : Très interprétable, coefficients explicables
- **Utilisation** : Quand la transparence est prioritaire

### Modèle éducatif : Arbre de décision
- **Avantage** : Parfaitement lisible, idéal pour montrer les règles apprises
- **Utilisation** : Explicabilité, formation, exploration

## Variables du modèle

### Entrées (Features)
- **province** : Province du Burundi (15 catégories)
- **culture** : Type de culture (6 catégories)
- **altitude_m** : Altitude moyenne (m)
- **pluviometrie_mm** : Pluviométrie totale de la saison (mm)
- **temperature_moy_C** : Température moyenne (°C)
- **superficie_ha** : Superficie cultivée (hectares)
- **utilisation_engrais** : Utilisation d'engrais (0/1)
- **acces_irrigation** : Accès à l'irrigation (0/1)

### Sortie (Target)
- **bonne_recolte** : Binaire (1=bonne récolte, 0=mauvaise récolte)
  - Définie comme : rendement > 85% du rendement moyen de la culture

## Données et déséquilibre

Le dataset contient **68% de bonnes récoltes** et **32% de mauvaises récoltes**. Cette classe imbalance est intentionnelle et reflète la réalité agricole au Burundi. Les modèles ont été évalués sur cette distribution réelle.

**Impact** : Le modèle tend à être plus optimiste (biais vers la classe majorité). En cas de faible pluviométrie, utiliser l'AUC plutôt que l'accuracy.

## Limitations et recommandations

### Limitations actuelles
1. **Données simulées** : Basées sur des données réelles agrégées, pas de mesures directes
2. **Déséquilibre de classe** : 68% de bonnes récoltes (biais optimiste du modèle)
3. **Absence de variables fines** : Texture du sol, fertilité, variétés, ravageurs
4. **Mémoire temporelle limitée** : Seulement 9 ans d'historique (2015-2023)
5. **Corrélation, pas causalité** : Le modèle apprend des corrélations, pas des causes

### Données supplémentaires utiles
- Texture et fertilité du sol
- Dose exacte d'engrais et calendrier d'apport
- Variété semée
- Dates de semis et récolte
- Intensité et répartition intra-saisonnière des pluies
- Fréquence d'irrigation
- Pression des ravageurs et maladies
- Pente et conditions de drainage
- Pertes post-récolte

## Support et questions

Pour toute question ou problème :
1. Consulter le `rapport.md` pour l'analyse technique détaillée
2. Examiner le notebook `TP_Agriculture_Burundi.ipynb` pour voir le pipeline complet
3. Ouvrir une issue sur le dépôt GitHub

## Crédits et contexte pédagogique

**TP Agriculture Burundi** - Université Polytechnique de Gitega (UPG)  
**Programme** : Bac 4 - Génie Logiciel  
**Module** : Intelligence Artificielle Appliquée

**Objectifs pédagogiques** :
- Découvrir le pipeline complet de Machine Learning (exploration → modélisation → déploiement)
- Comprendre les différences entre arbres, forêts et régression logistique
- Mettre en pratique la validation croisée, l'overfitting, les courbes ROC
- Déployer un modèle ML en production

## Licence et utilisation

Ce projet est fourni à des fins pédagogiques. Les modèles et données ne doivent pas être utilisés pour des décisions agricoles sans validation d'experts.

---

**Dernière mise à jour** : Mai 2026  
**Version** : 1.0 - Production Ready  
**Status** : ✅ Complètement testé et documenté
