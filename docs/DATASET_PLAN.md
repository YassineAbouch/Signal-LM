# Plan de données — Signal-LM

Objectif : disposer de données **traçables et fiables** pour entraîner plus tard
un spécialiste, sans présenter le modèle comme un expert. Ce document couvre
le lot « données » (branche `data-prep`) : il ne lance aucun entraînement et
ne touche ni à l'architecture ni aux poids vierges.

## 1. Périmètre et ordre

| Spécialiste | Priorité | Tâches préparées | Données v1 |
|---|---|---|---|
| Vibrations industrielles | **1** | détection d'événement, classification de motif, anomalie vs référence, multi-label | synthétique `synthetic_vibration_v1` (8 192 Hz, 1 s, 16 recettes) |
| Événements acoustiques simples | **1** | détection (début/fin), classification, anomalie, multi-label | synthétique `synthetic_acoustic_events_v1` (16 kHz, 1 s, 9 recettes) |
| Séries lentes (température, humidité, pression, dérives) | 2 | classification de motif, détection de changement, multi-label | synthétique `synthetic_slow_series_v1` (1 Hz, 1 h, 5 recettes) |
| Musique (instruments, polyphonie) | 3 | mono-instrument puis multi-label | aucune donnée synthétique (sans intérêt) ; NSynth / OpenMIC identifiés |

Chaque spécialité garde **son propre espace de classes** (`specialist` +
`task_type` dans le manifeste). Aucun jeu ne mélange les domaines.

Les quatre tâches ne sont pas interchangeables (SPEC_V1) :

- **Détection d'événement** : annotations `events` `[start, end)` ; une fenêtre sans événement porte `no_event`.
- **Classification d'événement / de motif connu** : un label par enregistrement.
- **Anomalie vs référence** : `reference` contre `deviation` ; le futur modèle apprend sur `reference` seul et le test contient les deux.
- **Multi-label** : plusieurs composantes présentes → plusieurs labels.

Les labels synthétiques sont **descriptifs** (`periodic_impulses`, `chirp`,
`damped_burst_short`) et jamais un diagnostic mécanique (pas de « défaut de
roulement confirmé »). Cette règle est reportée dans les `notes` de chaque jeu.

## 2. Ce que mesure chaque test

| Jeu | Partition | Ce que le test mesure |
|---|---|---|
| synthétique, split aléatoire par parent | test | généralisation à de **nouvelles réalisations** (nouveaux paramètres tirés dans les mêmes plages, nouveau bruit) |
| synthétique, règle `holdout` (`split_v1.json`) | test | généralisation à des **paramètres non vus** : sinus 1 000–1 500 Hz, oscillations amorties à décroissance 650–800 s⁻¹ jamais vues en train |
| synthétique, niveaux de SNR | test, par niveau | robustesse aux signaux faibles : évaluer par `snr_db_target` (−10 … 20 dB et propre), jamais en moyenne globale |
| réel (à venir), `group_id` = machine | test | généralisation à une **nouvelle machine** |
| réel, `group_id` = session / acquisition | test | généralisation à une **nouvelle acquisition** de machines connues |

Le test final ne sert ni au choix des seuils ni aux réglages : c'est le rôle de
`validation`.

## 3. Règles anti-fuite (implémentées dans `signal_lm.data.split`)

1. L'unité de séparation est `group_id`, fixée **avant** tout fenêtrage : parent synthétique, machine, session ou acquisition.
2. Toutes les variantes bruitées d'un même signal propre partagent le `group_id` → même partition.
3. Les fenêtres (`parent_record_id`) héritent de la partition du parent ; `window` refuse un manifeste non séparé.
4. Les règles `holdout` réservent des paramètres non vus pour le test.
5. Les augmentations d'entraînement se font après séparation (non implémentées dans ce lot ; le schéma les prévoit via `parent_record_id` + `transforms`).
6. La normalisation sera estimée sur `train` seulement (hors périmètre ici ; aucune normalisation n'est stockée dans les fichiers).
7. `check_leakage` détecte : partition non assignée, groupe réparti sur plusieurs partitions, fenêtre d'une autre partition que son parent, fichiers identiques (sha256) dans plusieurs partitions (les fenêtres constantes, par exemple vides d'événement, sont exemptées car identiques par contenu et non par origine).

## 4. Contrôles qualité (`signal_lm.data.quality`)

Par enregistrement : schéma, fichier présent, sha256, `n_samples` et nombre de
canaux cohérents avec le tableau, NaN/inf, canal constant, troncature (zéros
finaux, données réelles seulement), saturation (heuristique sur données réelles ;
sur synthétique, contrôle que l'écrêtage déclaré a bien eu lieu), bornes des
événements, SNR mesuré vs cible (±0,05 dB, parents seulement), fréquence et
amplitude dominantes mesurées par `signal_lm.analyze` vs paramètres générés
(sinus pur), régénération à l'identique depuis les paramètres (`--regenerate`).
Par jeu : doublons d'identifiant, fuites, équilibre des classes (ratio max/min
> 5 signalé). Le rapport `quality_report.{json,md}` liste volumes, classes,
durées, fréquences, SNR par cible et partitions.

Aucune fenêtre n'est normalisée : l'amplitude peut être un indice.

## 5. Format commun

Voir `docs/DATA_SCHEMA.md` : NPZ `[échantillons, canaux]` float32 + manifeste
JSONL. Import CSV (`signal_lm.load_csv`, colonnes et fréquence explicites) et
WAV (stdlib `wave`, PCM 8/16/24/32 bits, unité `unspecified`). Pas de
rééchantillonnage : chaque spécialité garde sa fréquence.

## 6. Premier lot synthétique — résultat

Généré hors Git (`data/`), reproductible (`seed` dans chaque config, graine par
parent dérivée par SHA-256), validé avec `--regenerate --strict` : **0 erreur**.

| Jeu | Parents | Fenêtres | Groupes train/val/test | Taille |
|---|---|---|---|---|
| `synthetic_vibration_v1` | 672 (16 recettes × 6 parents × 7 niveaux) | 4 704 (2 048 éch., pas 1 024) | 61 / 16 / 19 (3 groupes forcés en test par holdout) | ~160 Mo |
| `synthetic_acoustic_events_v1` | 378 (9 × 6 × 7) | 0 | 33 / 9 / 12 (3 holdout) | ~57 Mo |
| `synthetic_slow_series_v1` | 80 (5 × 4 × 4) | 0 | 10 / 5 / 5 | ~3 Mo |

SNR mesuré = cible à 10⁻⁹ dB près sur tous les enregistrements parents ; les
seuls avertissements restants sont 36 fenêtres constantes (fenêtres propres
sans impulsion, étiquetées `no_event`), attendues.

**Ce jeu vérifie le pipeline. Il ne prouve aucune performance industrielle.**

## 7. Limites et biais connus

- Signaux synthétiques stationnaires, un seul canal, durées fixes (1 s / 1 h) : pas de variation de vitesse, pas de transitoires de démarrage, pas de réponse de capteur.
- Bruit coloré 1/fᵅ seulement ; aucun bruit d'usine réel.
- Les labels de l'anomalie synthétique (`deviation`) sont construits par addition d'une composante : facile, pas représentatif d'une dérive réelle.
- Peu de parents par recette (4–6) : les holdouts ne touchent que 2–3 groupes ; augmenter `n_parents` avant toute expérience sérieuse.
- La fenêtre de 2 048 échantillons à 8 192 Hz (0,25 s) est un choix de vérification, pas une fenêtre validée par phénomène.
- Les sources réelles identifiées (`docs/DATA_SOURCES.md`) sont majoritairement des bancs de laboratoire ; plusieurs sont **non commerciales** (Paderborn, DCASE 2020, ESC-50) ou sans licence énoncée (CWRU).

## 8. Prochaine étape : brancher au pipeline d'entraînement

1. **Lecteur de jeu** : itérer `manifest.jsonl` filtré par `split`, `specialist`, `task_type` ; charger `mix` via `signal_lm.data.load_signal` (retourne un `Signal`, donc contrat d'entrée SPEC_V1 respecté) ; empiler en `[batch, samples, channels]` avec une longueur commune (fenêtrer avant, le CNN n'a pas de padding masqué).
2. **Cibles** : `labels` → index par (`specialist`, `task_type`) ; `events` → masques temporels pour la détection ; `reference`/`deviation` pour l'anomalie (entraînement sur `reference` seul).
3. **Normalisation** : statistiques calculées sur `train` uniquement et stockées dans le futur checkpoint spécialisé (SPEC_V1 « format spécialisé »), jamais dans les fichiers de données.
4. **Évaluation** : par partition, par `snr_db_target`, et par règle de holdout (lire `split_summary.json`) ; résultats à joindre au checkpoint avec `source_id`, `dataset_id`, `seed` et sha256 du manifeste pour la provenance.
5. **Données réelles** : importer le sous-ensemble recommandé (MIMII fan 0 dB, NSynth test, UCI Air Quality) avec `import-wav` / `import-csv`, `group_id` = machine ID, puis `split --config` avec `stratify_by: machine_id` et un holdout machine.

## 9. Reproduire sous Windows (PowerShell)

Depuis `E:\Signal-LM`, environnement déjà installé (README) :

```powershell
Set-Location E:\Signal-LM
$py = ".\.venv\Scripts\python.exe"

# Tests du pipeline de données (numpy seulement) + tests existants
& $py -m pytest -q

# Jeu synthétique vibration : génération, séparation, fenêtrage, validation
& $py -m signal_lm.data.cli generate --config configs\data\synthetic_vibration_v1.json --root data\synthetic_vibration_v1
& $py -m signal_lm.data.cli split    --root data\synthetic_vibration_v1 --config configs\data\split_v1.json
& $py -m signal_lm.data.cli window   --root data\synthetic_vibration_v1 --window-samples 2048 --hop-samples 1024
& $py -m signal_lm.data.cli validate --root data\synthetic_vibration_v1 --regenerate --strict

# Événements acoustiques et séries lentes
& $py -m signal_lm.data.cli generate --config configs\data\synthetic_acoustic_events_v1.json --root data\synthetic_acoustic_events_v1
& $py -m signal_lm.data.cli split    --root data\synthetic_acoustic_events_v1 --config configs\data\split_v1.json
& $py -m signal_lm.data.cli validate --root data\synthetic_acoustic_events_v1 --regenerate --strict
& $py -m signal_lm.data.cli generate --config configs\data\synthetic_slow_series_v1.json --root data\synthetic_slow_series_v1
& $py -m signal_lm.data.cli split    --root data\synthetic_slow_series_v1 --config configs\data\split_v1.json
& $py -m signal_lm.data.cli validate --root data\synthetic_slow_series_v1 --regenerate --strict

# Résumé
& $py -m signal_lm.data.cli show --root data\synthetic_vibration_v1
Get-Content data\synthetic_vibration_v1\quality_report.md
```

Le script `scripts\prepare_synthetic_data.ps1` enchaîne ces commandes.
`--overwrite` est nécessaire pour régénérer dans un dossier non vide ; `data\`
est ignoré par Git. Import d'un WAV réel :

```powershell
& $py -m signal_lm.data.cli import-wav C:\datasets\mimii\fan\id_00\normal\00000000.wav `
    --root data\mimii_fan_0db --record-id mimii_fan_0db__id00__normal_00000000 `
    --source-id mimii_fan_0db --specialist acoustic --task-type anomaly_detection `
    --group-id fan_id00 --machine-id fan_id00 --labels normal `
    --license "CC BY-SA 4.0 (https://zenodo.org/records/3384388)"
```
