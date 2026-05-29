# TP Agriculture Burundi

Projet complet d'IA appliquee a l'agriculture au Burundi.

## Contenu

- `TP_Agriculture_Burundi.ipynb` : notebook complet du TP
- `streamlit_app.py` : application web de prediction
- `model_arbre.pkl`, `model_foret.pkl`, `model_regression.pkl`, `scaler.pkl`, `feature_columns.pkl` : artefacts d'entrainement
- `rapport.md` : reponses aux questions de reflexion

## Lancer l'application en local

1. Installer les dependances :

```bash
pip install -r requirements.txt
```

2. Demarrer Streamlit :

```bash
streamlit run streamlit_app.py
```

## Deploiement sur Streamlit Community Cloud

### Option la plus simple : sans Git

1. Creer un depot GitHub **public** vide.
2. Ajouter les fichiers a la racine du depot :
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
3. Ouvrir Streamlit Community Cloud et cliquer sur **Create app**.
4. Choisir le depot GitHub, la branche `main` et le fichier principal `streamlit_app.py`.
5. Lancer le deploiement et attendre la fin du build.

### Option terminal : avec Git

Si `git` est installe sur la machine :

```bash
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/TON_UTILISATEUR/TON_DEPOT.git
git push -u origin main
```

Ensuite, connecter ce depot a Streamlit Community Cloud et garder `streamlit_app.py` comme fichier principal.

### Points a verifier

- `streamlit_app.py` doit etre a la racine du depot.
- Les fichiers `.pkl` et `agriculture_burundi.csv` doivent aussi etre a la racine.
- Si le deploiement echoue, ouvrir les logs Streamlit : l'erreur indique souvent un fichier manquant ou une version de package a corriger.

## Notes

- Le modele selectionne affiche la prediction, la probabilite, l'accuracy, le F1-score et l'AUC.
- L'application est concue pour accepter les champs du TP : province, culture, altitude, pluviometrie, temperature, superficie, engrais et irrigation.
- La culture `Maïs` est convertie en interne vers la valeur presente dans le fichier source (`Maļs`) pour rester compatible avec le dataset.
