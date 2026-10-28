# Premier spécialiste — cadrage et protocole proposé

## Statut

Document de préparation du lot 2. Le lot 1 reste un encodeur vierge et des
mesures déterministes. Aucun entraînement, import de données réelles ou choix
de classes n'est effectué par ce document. Le domaine retenu est « autres signaux
industriels ». Le signal précis et les critères métier restent à choisir avec
le responsable du projet.

## Fiche à compléter avant l'implémentation métier

| Décision | Valeur actuelle | Information nécessaire |
| --- | --- | --- |
| Domaine | Autres signaux industriels | Choix utilisateur ; signal précis à définir |
| Signal prioritaire | À définir | Par exemple courant, pression, température, débit ou couple |
| Usage | À définir | Décision attendue, utilisateur et conséquences d'une erreur |
| Objectif appris | À choisir | Classification exclusive, multi-label ou anomalie |
| Classes / état de référence | Non définis | Définition observable et provenance des étiquettes |
| Acquisition | Non définie | Capteur, montage, machine, régime, période et synchronisation |
| Entrée | Contrat V1 | Fréquence, canaux, unités, durée et bande utile |
| Généralisation visée | À définir | Nouvelle acquisition, période future ou nouvelle machine |
| Contraintes d'exploitation | À définir | Latence, mémoire, fréquence des décisions et matériel |
| Acceptation métier | À fixer avant le test final | Erreurs tolérées et couverture minimale |

Ne pas choisir un nombre de classes ni une durée de fenêtre arbitraires pour
remplir cette fiche. La durée devra permettre d'observer les phénomènes visés ;
le champ réceptif local actuel est limité à 19 échantillons avant pooling.

## Protocole de données proposé

1. Décrire les acquisitions dans un manifeste local, avec un identifiant stable,
   un identifiant de machine, une période, les conditions de fonctionnement,
   la fréquence, les canaux, les unités, les étiquettes et leur provenance.
   Les chemins sensibles et identifiants réels restent hors Git.
2. Définir l'unité de séparation selon l'usage : machine si l'objectif est une
   machine inconnue ; acquisition pour une nouvelle acquisition ; période
   pour une prévision sur le même parc. Éviter les recouvrements temporels et
   les copies d'un même enregistrement entre partitions.
3. Affecter les groupes à train / validation / test **avant** de créer les fenêtres.
   Ne jamais répartir aléatoirement des fenêtres voisines d'une acquisition
   entre partitions. Documenter les proportions après inventaire des groupes,
   sans imposer un ratio qui rendrait certaines classes absentes.
4. Vérifier les groupes disjoints, les doublons, les valeurs finies, les unités,
   la régularité temporelle et la distribution des classes par partition.
   Le lecteur CSV V1 ne contrôle pas les timestamps : ce contrôle devra être
   ajouté explicitement au futur pipeline si des timestamps sont fournis.
5. Ajuster normalisation et prétraitements appris exclusivement sur train.
   Appliquer ensuite les mêmes paramètres figés à validation et test.
   Conserver les mesures FFT/RMS sur les valeurs physiques indépendamment.
6. Appliquer toute augmentation uniquement à train. Conserver une filiation
   vers l'acquisition source pour les fenêtres et leurs variantes.

## Évaluation proposée

Le choix du modèle, des hyperparamètres et des seuils utilise train et
validation. Réserver à la calibration un sous-ensemble distinct de celui qui
sert à sélectionner le modèle, si le volume de groupes le permet. Si les
données sont insuffisantes, déclarer la calibration non validée et réduire la
portée des conclusions. Garder le test final intact jusqu'au gel des décisions.

| Objectif retenu | Résultats à produire |
| --- | --- |
| Classification exclusive | Matrice de confusion, précision/rappel par classe, macro-F1, effectifs par classe |
| Multi-label | Précision/rappel/F1 par étiquette, seuils par étiquette choisis hors test |
| Anomalie | Rappel des événements, faux positifs par durée d'observation, délai de détection ; référence normale définie |

Rapporter les résultats par machine, acquisition, régime et niveau de bruit
lorsque ces informations existent. Estimer l'incertitude au niveau des groupes
indépendants plutôt qu'en traitant les fenêtres corrélées comme indépendantes.
Comparer à une référence simple définie à l'avance, par exemple une règle sur
les mesures physiques ou un classifieur simple sur ces mesures.

Si une confiance est exposée, évaluer sa calibration sur des données réservées.
Un rejet « inconnu » nécessite des cas hors périmètre et des seuils validés ;
une softmax seule ne suffit pas. Les seuils de réussite métier restent à fixer
avant toute consultation du test final. Aucune précision cible n'est revendiquée.

## Livrables d'implémentation à venir

1. Manifeste et contrôles de séparation, vérifiés sur des acquisitions synthétiques.
2. Fenêtrage avec provenance et normalisation ajustée uniquement sur train.
3. Configuration de spécialité explicitant l'entrée, la tâche et les classes.
4. Boucle d'apprentissage reproductible et format de checkpoint spécialisé
   distinct du format vierge, avec prétraitement, provenance et évaluation.
5. Rapport d'évaluation reproductible, puis mesure de latence et mémoire sur
   le poste cible avec la durée de fenêtre retenue.

Les deux premiers livrables peuvent être développés et testés sur des données
synthétiques sans entraîner de spécialiste. Le démarrage d'un entraînement
métier attendra une instruction explicite et les choix de la fiche ci-dessus.
