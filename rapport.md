# Rapport d'Analyse - TP Agriculture Burundi

## Résumé exécutif

Ce rapport présente l'analyse approfondie du projet de prédiction des récoltes au Burundi basée sur trois modèles de Machine Learning : Arbre de Décision, Forêt Aléatoire et Régression Logistique. Le modèle recommandé pour déploiement en production est la **Forêt Aléatoire** en raison de sa robustesse supérieure (93.67% accuracy, validation croisée stable).

---

## Q29 - Scénario de Gitega (Haricot avec faible pluviométrie)

### Contexte
Pour le scénario **Gitega - Haricot** avec conditions difficiles :
- Altitude : 1 720 m
- Pluviométrie : **430 mm** (très faible)
- Température : 18.2 °C
- Engrais : Non utilisé
- Irrigation : Non disponible

### Prédictions des trois modèles

| Modèle | Prédiction | Probabilité | Confiance |
|--------|-----------|-------------|----------|
| Arbre de décision | Bonne récolte | 0.82 | Modérée |
| Forêt aléatoire | Bonne récolte | 0.85 | Modérée |
| Régression logistique | Bonne récolte | 0.84 | Modérée |

### Analyse critique

Tous les modèles prédisent une **bonne récolte**, mais cette prédiction doit être interprétée avec prudence :

**Résultat paradoxal** : Malgré une pluviométrie extrêmement basse (430 mm), les trois modèles sont optimistes. Cela révèle une limitation importante du modèle :

1. **Biais du dataset** : Le jeu de données contient 68% de bonnes récoltes vs 32% de mauvaises. Le modèle apprend à être optimiste par défaut.

2. **Couverture du dataset** : La pluviométrie 430 mm peut être un cas rare dans le dataset original. Le modèle extrapolé en conditions extrêmes manque d'exemples réels.

3. **Interactions non capturées** : Le stress hydrique combiné à l'absence d'irrigation pourrait avoir un effet multiplicatif (synergique) que le modèle ne capture pas convenablement.

### Recommandations agronomiques pour Gitega (Haricot)

**⚠️ Attention** : Une probabilité de 0.84 ne signifie PAS une certitude de 84%. En conditions de stress hydrique extrême, mettre en place :

1. **Gestion de l'eau (priorité absolue)**
   - Installer un système de récolte d'eau pour pallier le déficit
   - Appliquer du paillage pour conserver l'humidité du sol
   - Couvrir le sol avec des débris organiques
   - Envisager une source d'irrigation de secours (forage, citernes)

2. **Sélection variétale**
   - Choisir des variétés de haricot tolérantes à la sécheresse
   - Privilégier les lignées locales adaptées aux conditions semi-arides
   - Tester en parcelles pilotes avant déploiement massif

3. **Calendrier agricole**
   - Ajuster la date de semis pour coïncider avec les épisodes de pluie prévus
   - Consulter les prévisions météorologiques avant semis
   - Planter en anticipant la distribution intra-saisonnière des pluies

4. **Suivi et intervention précoce**
   - Surveiller étroitement le bilan hydrique dès le semis
   - Intervenir précocement en cas de sécheresse (irrigation d'appoint ou paillage renforcé)
   - Documenter les rendements réels pour améliorer les prédictions futures

5. **Engrais**
   - L'absence d'engrais limite aussi le rendement. Envisager un apport minimal si les ressources le permettent
   - Même en conditions sèches, un peu d'engrais peut améliorer l'efficacité d'utilisation de l'eau

---

## Q30 - Recommandation pour le Ministère de l'Agriculture

### Quel modèle déployer en production ?

**Recommandation : Forêt Aléatoire**

#### Justification

| Critère | Forêt Aléatoire | Régression Logistique | Arbre de Décision |
|---------|-----------------|--------------------|--------------------|
| **Accuracy test** | **0.9367** ⭐ | 0.8903 | 0.9245 |
| **AUC** | 0.8155 | **0.8458** ⭐ | 0.7842 |
| **Validation croisée** | **0.9341 ± 0.0032** ⭐ (très stable) | 0.8624 ± 0.0156 | 0.9015 ± 0.0087 |
| **Robustesse** | **Excellente** ⭐ | Bonne | Moyenne |
| **Interprétabilité** | Moyenne | **Excellente** ⭐ | **Excellente** ⭐ |
| **Temps d'inférence** | Rapide | Très rapide | Très rapide |

**Résultat** : La **Forêt Aléatoire** offre le meilleur compromis :
- **Précision maximale** : 93.67% accuracy
- **Stabilité exceptionnelle** : écart-type de 0.32% sur 5 folds (quasi pas de variance)
- **Relations non-linéaires** : capture mieux les interactions complexes entre climat, sol et pratiques agricoles
- **Robustesse** : résiste mieux à l'overfitting et aux nouvelles données

### Rôle complémentaire de la Régression Logistique

La **Régression Logistique** doit être conservée comme modèle **secondaire pour la transparence** :
- **Meilleure AUC** (0.8458) : montre une meilleure capacité de discrimination des risques
- **Coefficients interprétables** : chaque variable a un coefficient explicable (ex: "+pluviométrie → +récolte")
- **Usage** : Quand la transparence et la justification sont prioritaires pour les décideurs politiques

### Déploiement recommandé

**Architecture hybrid** :
1. **Production** : Forêt Aléatoire (pour prédictions opérationnelles)
2. **Justification** : Régression Logistique (pour expliquer les décisions aux agriculteurs)
3. **Monitoring** : Arbre de Décision (pour visualiser les règles apprises en temps réel)

---

## Données supplémentaires pour améliorer les prédictions

### Facteurs pédologiques (sol)
- Texture du sol (argile, limon, sable %)
- Teneur en matière organique / humus
- Fertilité et phosphore disponible (P-disponible)
- pH du sol
- Capacité de rétention d'eau (CRE)
- Profondeur du sol

### Facteurs agronomiques précis
- **Engrais** : dose exacte (kg/ha), type (NPK), calendrier d'apport
- **Variété semée** : référence variétale, précocité, tolérance à la sécheresse
- **Calendrier** : dates exactes de semis et récolte
- **Irrigation** : fréquence (jours), dose (mm), méthode (goutte-à-goutte, gravitaire)

### Facteurs climatiques détaillés
- **Pluviométrie intra-saisonnière** : distribution jour par jour, longueur de la période sèche
- **Température** : min/max quotidiens, jours avec stress thermique
- **Humidité** : humidité relative quotidienne
- **ETP (Evapotranspiration Potentielle)** : pour calculer le bilan hydrique réel

### Facteurs biologiques
- **Ravageurs et maladies** : type, intensité, période d'infection
- **Adventices** : densité, concurrence
- **Microbes du sol** : champignons bénéfiques, nématodes parasites

### Données économiques et sociales
- **Coût des intrants** : prix de l'engrais, de l'eau, semences
- **Main-d'œuvre** : disponibilité, coût des opérations
- **Infrastructure** : accès au crédit, à des marchés, à des informations
- **Pratiques antérieures** : historique des rendements de la parcelle

---

## Limites du système actuel

### 1. Données et représentativité
- ❌ **Données simulées**, pas mesurées in situ
- ❌ Basées sur agrégation au niveau provincial/saisonnier, perte de granularité
- ❌ Couvre seulement **9 années** (2015-2023) : insuffisant pour cycles long terme ou changement climatique
- ❌ **6 cultures seulement** : ignore la rotation, l'agroforesterie, les cultures de rente

### 2. Déséquilibre et biais
- ❌ **68% bonnes récoltes vs 32% mauvaises** : biais optimiste structurel
- ❌ Modèle plus prudent sur les risques que sur les succès attendus
- ❌ En cas de stress extrême, confiance trop élevée

### 3. Variables manquantes
- ❌ Aucune variable agronomique fine (dose d'engrais, variété, ravageurs)
- ❌ Pas de facteurs pédologiques (texture, fertilité du sol)
- ❌ Pas de pratiques culturales détaillées (semis dense/clairsemé, entretien)

### 4. Causalité vs corrélation
- ❌ **Le modèle apprend des corrélations**, pas des causes
- ❌ Exemple : Si dans le dataset, les bonnes récoltes coïncident avec une région donnée, le modèle apprendra ça, même si la vraie cause est ailleurs (meilleur sol, meilleures pratiques)

### 5. Décisions irrévocables
- ❌ Malgré 93% accuracy, le modèle se trompe sur **1 parcelle sur 11**
- ❌ En agriculture, une mauvaise décision (semis sans irrigation en année sèche) est **coûteuse et irréversible**

---

## Recommandations d'usage en production

### ✅ FAIRE
1. **Utiliser le modèle comme outil d'aide à la décision**, jamais comme verdict définitif
2. **Combiner avec l'expertise locale** : consultant agricole + modèle ML
3. **Expliquer les prédictions** : montrer les facteurs dominants (importance des variables)
4. **Collecter les retours** : documenter les cas où le modèle s'est trompé pour améliorer
5. **Mettre à jour** : ré-entraîner chaque année avec les données réelles nouvelles

### ❌ NE PAS FAIRE
1. ❌ Faire confiance aveuglément au modèle en cas de stress extrême
2. ❌ Ignorer l'expertise des agriculteurs locaux
3. ❌ Utiliser le modèle sans avoir consulté une personne qualifiée
4. ❌ Généraliser au-delà des provinces/cultures du dataset
5. ❌ Supposer que les relations apprises en 2015-2023 resteront valides pendant 10 ans

---

## Conclusion

La **Forêt Aléatoire** atteint une performance exceptionnelle (93.67%) et reste stable en validation croisée. Cependant, les rendements agricoles dépendent de nombreux facteurs non capture par le dataset actuel.

**Le modèle doit être vu comme un capteur d'alerte**, pas une boule de cristal. Son rôle est de :
1. Identifier les situations à **haut risque** (mauvaise récolte probable)
2. Justifier pourquoi le risque est élevé
3. Proposer des **pistes d'action** pour l'atténuer

Avec une collecte de données plus riche et un engagement des agronomes locaux, ce système pourrait devenir un outil décisionnel stratégique pour les ministères agricoles du Burundi et de la région.

---

## Références techniques

- **Validation** : 5-fold Stratified Cross-Validation
- **Données d'entraînement** : 1 296 observations (80%)
- **Données de test** : 324 observations (20%)
- **Période couverte** : 2015-2023 (9 années)
- **Zones** : 15 provinces du Burundi
- **Cultures** : Maïs, Haricot, Manioc, Patate douce, Sorgho, Bananier
