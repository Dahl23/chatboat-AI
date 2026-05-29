# Rapport - TP Agriculture Burundi

## Q29 - Scenario de Gitega

Pour le scenario `Gitega - Haricot`, les trois modeles prevoient une bonne recolte, avec toutefois la probabilite la plus faible pour Gitega parmi les quatre scenarios testes. Les probabilites de bonne recolte restent superieures a 0.82, mais le contexte agronomique est plus delicat car la pluviometrie est faible (430 mm) et l'irrigation est absente.

Cette prediction n'est donc que partiellement rassurante. Le modele peut sous-estimer le risque lie au stress hydrique, notamment parce que le jeu de donnees est fortement desequilibre vers la classe `bonne_recolte = 1`.

Recommandations pratiques pour Gitega :

- renforcer la gestion de l'eau (irrigation, collecte d'eau, paillage, couverture du sol),
- choisir des varietes plus tolerantes a la secheresse,
- ajuster la date de semis pour mieux coller aux episodes de pluie,
- suivre de pres l'etat hydrique et intervenir precocement si la saison devient seche.

## Q30 - Recommandation pour le Ministere de l'Agriculture

Je recommanderais de deployer la **Foret aleatoire** en production. C'est le meilleur compromis entre performance et robustesse : elle obtient la meilleure accuracy de test (0.9367) et une validation croisee tres stable (0.9341 +/- 0.0032). Elle capture aussi mieux les relations non lineaires entre climat, pratiques agricoles et rendement.

La regression logistique reste interessante pour l'interpretablite, et elle obtient la meilleure AUC (0.8458), ce qui montre une bonne capacite de classement des risques. Elle pourrait donc servir de modele complementaire lorsque la transparence est prioritaire.

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
