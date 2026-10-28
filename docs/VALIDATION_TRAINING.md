# Validation du jalon 2 — 28 octobre 2026

## Périmètre

Branche `data-prep`, lot données de Claude Code et travail audio de `main`
réunis sur cette branche. Aucun changement de générateur, du schéma des
manifestes ou du modèle vierge. La variante temporelle et le format spécialisé
sont spécifiés dans `SPEC_V1.md`. Procédure : `TRAINING.md`.

Le dossier de travail contient aussi des ajouts de fusion multi-capteurs en
cours, indépendants de ce lot. Pour vérifier précisément le code livré, le
contenu préparé dans l'index Git a été exporté sous `artifacts/index-validation-20261028`
et testé avec son propre dossier `src` dans `PYTHONPATH` : **100 tests réussis
en 16,59 s**. Aucun module fusion non commité n'était nécessaire à ces tests.

## Données synthétiques

`scripts/prepare_synthetic_data.ps1` exécuté sans écrasement des données.
La première exécution a exposé une reprise non idempotente et une erreur de
régénération des fenêtres. Corrections limitées au script et au contrôle qualité :
reprise sur manifestes existants, arrêt sur code d'erreur, comparaison des fenêtres
aux tranches des parents ; aucune modification des signaux stockés.

| Jeu | Parents | Fenêtres | Erreurs | Avertissements |
| --- | ---: | ---: | ---: | ---: |
| synthetic_vibration_v1 | 672 | 4 704 | 0 | 37 |
| synthetic_acoustic_events_v1 | 378 | 0 | 0 | 1 |
| synthetic_slow_series_v1 | 80 | 0 | 0 | 0 |

Les fenêtres constantes attendues sont signalées ; un écart d'un ULP float32
dans le bruit acoustique est désormais un avertissement explicite d'arrondi,
pas une revendication de régénération bit à bit. Les différences plus grandes
restent des erreurs. Les fichiers et générateurs n'ont pas été réécrits.

## Vérification d'apprentissage et d'évaluation

Environnement Windows, Python 3.13.2, PyTorch 2.13.0+cu130, NumPy 2.5.4,
RTX 4070 Laptop. Une époque par tâche sur `synthetic_vibration_v1`, fenêtres
de 2 048 échantillons, batch 32, seed 42, AdamW lr 0,001. Les quatre essais
utilisent le même protocole fixé avant de consulter le test. Les checkpoints
et résultats détaillés restent sous `artifacts/training/` et `artifacts/`.

| Tâche | Étapes du checkpoint | Fenêtres test | Macro-F1 test |
| --- | ---: | ---: | ---: |
| event_classification | 28 | 343 | 0,0715 |
| event_detection | 23 | 245 | 0,0849 |
| multilabel_classification | 25 | 196 | 0,1429 |
| anomaly_detection | 7 | 147 | 0,1217 |

Ces scores faibles sont des résultats de vérification de fonctionnement sur
une époque, pas une démonstration de compétence. La détection est mesurée au
niveau échantillon. Aucune itération de seuil ou d'hyperparamètre n'a utilisé
ces scores test. Les limites du champ réceptif, de la baseline une-classe et
du nombre de groupes restent explicites dans `SPEC_V1.md` et `TRAINING.md`.

Les tests vérifient pour les quatre tâches : aucun accès aux tableaux test
pendant `fit`, cibles et formes cohérentes, statistiques train seules,
références seules pour l'anomalie, seuil ajusté sur références validation,
checkpoints rechargés avec sorties CPU exactes et refus d'écrasement. La copie
d'évaluation conserve exactement poids, normalisation et seuil. Les résultats
sont ventilés par SNR cible, holdout et machine. Un changement de manifeste,
config ou résumé de split empêche l'évaluation avec une provenance incompatible.

## MIMII fan à 0 dB

Source officielle : https://zenodo.org/records/3384388, public 1.0,
Purohit et al., CC BY-SA 4.0. Archive complète téléchargée :
`data/downloads/0_dB_fan.zip`, 10 411 902 283 octets. MD5 vérifié :
`6354d1cc2165c52168f9ef1bcd9c7c52`. Elle contient 5 550 WAV (14,21 Go décompressés).

Le premier sous-ensemble importé contient **200 WAV**, 25 par état et par machine,
sélectionnés par nom avant évaluation. Import via la fonction existante `import_wav` :
8 canaux, 16 kHz, unités `unspecified`, hashes originaux et licence conservés.
Sous `data/mimii_fan_0db/` :

- Train : machines 00 et 02, 100 clips.
- Validation : machine 04, 50 clips.
- Test : machine 06, 50 clips.
- Total : 100 `normal`, 100 `abnormal` ; `group_id = machine_id`.
- `validate --strict` : **0 erreur et 0 avertissement** ; absence de fuite vérifiée.

Le pilote conserve des machines entières par split, sans stratification à
l'intérieur d'une machine. Avec seulement quatre machines et une sélection
par noms de fichiers, il ne permet pas de conclure sur la généralisation terrain.
Le reste de l'archive est disponible, mais n'est pas importé dans ce pilote.
L'import complet commencé avant le choix du pilote est préservé séparément
dans `data/mimii_fan_0db_partial_import`, non validé pour entraînement.

**Aucun entraînement sur données réelles dans ce lot d'intégration.** Le pipeline refuse ce fit sans
l'option explicite `--allow-real-training`. Aucun de ces fichiers ou poids
n'est ajouté à Git. L'interface audio continue d'utiliser ses mesures déterministes.
Le premier entraînement réel autorisé ultérieurement est documenté séparément
dans [REAL_TRAINING_V1.md](REAL_TRAINING_V1.md).

## M4A dans l'interface audio

M4A/AAC standard, métadonnées en fin de fichier et fenêtre décalée testés.
Le serveur audio a été redémarré avec le décodeur à lecture aléatoire en mémoire.
Test HTTP sur le serveur effectif : `qa-tone.m4a` synthétique, 48 kHz,
fréquence dominante 440 Hz. Aucun fichier audio utilisateur n'a été conservé.

## Prochaines étapes

1. Relire le protocole et définir une référence simple ainsi que les métriques
   d'acceptation avant tout réglage supplémentaire.
2. Fenêtrer le pilote MIMII, puis décider explicitement d'une expérience réelle
   sur références train seules ; garder la machine test réservée.
3. Ajouter si nécessaire une calibration indépendante et des métriques par
   événement/durée ; aucune confiance métier n'est disponible à ce stade.
