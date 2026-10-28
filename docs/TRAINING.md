# Pipeline spécialisé — jalon 2

Travail sur `data-prep`, à partir du lot données de Claude Code. Les générateurs,
le schéma et le modèle vierge ne sont pas modifiés. Les fichiers de données et
les checkpoints restent hors Git. Aucun modèle de langage ni diagnostic mécanique
n'est produit par ce pipeline.

## Préparer et vérifier

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\prepare_synthetic_data.ps1
```

Le script réutilise les jeux existants sans les régénérer ou doubler les fenêtres,
et s'arrête si une commande échoue. `-Overwrite` reste une demande explicite de
régénération. La validation des fenêtres compare leurs tableaux aux tranches du
parent ; les parents sont régénérés avec les générateurs existants. Un écart
d'au plus un ULP float32 est signalé comme avertissement d'arrondi, jamais comme
une identité bit à bit. Une différence supérieure reste une erreur.

## Entraîner une seule spécialité et une seule tâche

Exemple d'essai d'intégration synthétique, une époque :

```powershell
.\.venv\Scripts\python.exe -m signal_lm.training.cli fit `
  --root data/synthetic_vibration_v1 --specialist vibration `
  --task-type event_classification --level windows --epochs 1 `
  --batch-size 32 --device cuda --output artifacts/training/classifier.pt
```

Options de tâche : `event_classification`, `multilabel_classification`,
`event_detection`, `anomaly_detection`. Les variantes sont décrites dans
`SPEC_V1.md`. `--level windows` sélectionne uniquement les fenêtres ; `parents`
sélectionne uniquement les parents, par exemple pour le jeu acoustique qui n'a
pas encore été fenêtré. Aucun padding ni rééchantillonnage n'est effectué.
Des longueurs, fréquences, noms de canaux ou unités différentes sont refusés.

La lecture passe par `signal_lm.data.load_signal(root, record)` puis empile les
exemples en `[batch, samples, channels]`. L'intégrité des NPZ est vérifiée avant
chargement ; les groupes et relations parent/enfant doivent rester dans un seul
split. Les doublons inter-splits sont refusés, sauf fenêtres synthétiques propres
`no_event` réellement constantes, contrôlées au chargement.

Les labels sont indexés à partir de train seulement, séparément pour chaque
couple spécialité/tâche. Une classe nouvelle en validation ou test entraîne une
erreur, pas une nouvelle classe ajoutée après coup. La détection temporelle
utilise les annotations `[start_sample, end_sample)` et des masques indépendants
par label ; `no_event` est un masque tout nul. Le multi-label encode la présence
de chaque label par 0/1. L'anomalie reconnaît explicitement `reference`/`normal`
et `deviation`/`abnormal`, sans transformer ces labels en diagnostic de défaut.

## Normalisation et choix du modèle

Moyenne et écart-type par canal sont ajustés sur les exemples train sélectionnés,
pondérés par le nombre d'échantillons. Les fenêtres qui se recouvrent comptent
donc plusieurs fois les échantillons partagés : c'est un choix de la population
d'entraînement, à conserver lors des comparaisons. Un canal constant utilise une
échelle de 1 et reste signalé dans le checkpoint. Aucun NPZ n'est réécrit.

Pour l'anomalie, le fit des statistiques, du centre et des poids utilise les
références train seules. Les déviations train ne participent pas à l'optimisation.
La variante une-classe par distance à un centre est une baseline minimale ; des
embeddings qui s'effondrent ou un score sans intérêt restent possibles. Comparer
à des références simples avant d'envisager une utilisation métier.

Les meilleurs poids sont sélectionnés par perte validation (références seules
pour l'anomalie). Le seuil de déviation est le quantile 0,95 fixé par défaut des
scores des références validation ; `--reference-quantile` permet une valeur
fixée avant évaluation finale. Les seuils multi-label/détection sont fixés à 0,5.
Ces procédures ne constituent pas une calibration indépendante de confiance.

`fit` ne charge aucun tableau test. Les graines et la configuration sont stockées ;
une identité numérique entre versions, pilotes ou CPU/GPU n'est pas garantie.
Les tests de relecture exacte sont effectués sur CPU dans un même environnement.

## Évaluation finale sans réajustement

```powershell
.\.venv\Scripts\python.exe -m signal_lm.training.cli evaluate `
  --root data/synthetic_vibration_v1 --checkpoint artifacts/training/classifier.pt `
  --splits test --device cuda --output artifacts/training/classifier_evaluated.pt
```

La commande crée une nouvelle copie avec résultats et empreinte du checkpoint
parent. Elle conserve les poids, le centre, la normalisation et les seuils.
Les empreintes du manifeste, de `config.json` et de `split_summary.json` doivent
correspondre à celles du fit. Les fichiers existants ne sont pas écrasés.
Ne pas réutiliser le test pour choisir une autre époque, un seuil ou une architecture.

Les métriques sont regroupées par split, par SNR cible (`clean`, −10/−5/0/5/10/20 dB
selon les données), par règle de holdout issue de `split_summary.json` et par
machine. Un SNR réel inconnu est `unknown`, distinct d'un synthétique propre.
Classification : matrice de confusion, exactitude, précision/rappel/F1 par classe.
Multi-label/anomalie : comptes TP/FP/FN et F1. Détection : mêmes comptes au niveau
échantillon ; aucun résultat de délai, de segmentation d'événements ou de faux
positifs par heure n'est revendiqué. Les fenêtres corrélées ne sont pas des
observations indépendantes ; le rapport inclut les effectifs de groupes, sans
intervalle d'incertitude dans cette première version.

## Checkpoint spécialisé

`signal_lm.training.engine.load_specialist(path)` recharge le modèle et les
métadonnées sur CPU avec `weights_only=True`. Le format est distinct du format
vierge : `signal_lm.specialist.v1`, statut `trained`, labels/index, contrat
d'entrée, normalisation, configuration et étapes, poids, centre/seuil, historique
de sélection, dataset_id, dataset_seed, training_seed, sources/licences, SHA-256
du manifeste/config/splits et évaluations. `load_virgin` refuse ce checkpoint.
Le statut `trained` atteste des étapes effectuées, pas d'une compétence validée.

## Premier import réel : MIMII fan 0 dB

Source officielle : https://zenodo.org/records/3384388, public 1.0, licence
CC BY-SA 4.0 ; attribution à Purohit et al. (DCASE 2019) et obligations de la
licence à conserver pour toute redistribution. Archive `0_dB_fan.zip`,
10 411 902 283 octets, MD5 officiel `6354d1cc2165c52168f9ef1bcd9c7c52`.

```powershell
.\.venv\Scripts\python.exe scripts/download_mimii.py
.\.venv\Scripts\python.exe scripts/import_mimii_fan.py
.\.venv\Scripts\python.exe -m signal_lm.data.cli validate --root data/mimii_fan_0db --strict
```

Le téléchargement est reprenable par parties et vérifié avant import. Les parties
restent sous `data/downloads/`, pour permettre une reprise et éviter toute
suppression automatique. L'import utilise la fonction existante `import_wav`
(même lecteur que `import-wav`), conserve les 8 canaux, 16 kHz et unités
`unspecified`, le hash original, l'entrée ZIP, le DOI et la licence.

Le premier lot est un **pilote de 200 WAV** : les 25 premiers fichiers triés
par nom pour chaque couple machine/état, sur 4 machines. Ce choix est déterminé
avant toute évaluation et comporte un biais potentiel lié aux noms de fichiers.
Ce n'est pas l'import complet des 5 550 WAV de l'archive. Pour étendre plus tard :
`--max-per-state 0 --root data/mimii_fan_0db_full` (nouveau dossier explicite).
Les imports existants ne sont pas remplacés.

`group_id = machine_id` ; machines 00 et 02 en train, 04 en validation, 06 en test.
Une stratification à l'intérieur de chaque machine serait contradictoire avec
ce groupement et provoquerait une fuite ; le split conserve des machines entières.
Les nombres normal/anormal sont équilibrés dans chaque machine du pilote.
Les proportions effectives sont donc 50 % / 25 % / 25 % des enregistrements,
avec seulement 2 / 1 / 1 machines : généralisation très peu documentée à ce stade.

L'import n'entraîne rien. Le fit réel est refusé sans `--allow-real-training`.
Il faudra d'abord fenêtrer ce jeu (`window`) et définir l'expérience. Aucun
entraînement réel n'a été autorisé ou réalisé dans ce lot de vérification.
Un import complet partiel antérieur a été conservé dans
`data/mimii_fan_0db_partial_import` ; il n'est pas un jeu prêt à entraîner.
