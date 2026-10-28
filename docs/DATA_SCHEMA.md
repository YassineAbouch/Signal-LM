# Schéma de données — Signal-LM (v1)

Un jeu de données est un dossier (hors Git, sous `data/`) :

```
data/<dataset_id>/
  config.json            configuration de génération ou d'import qui l'a produit
  manifest.jsonl         un enregistrement JSON par ligne (schéma ci-dessous)
  records/<record_id>.npz  tableaux [échantillons, canaux] float32
  split_summary.json     écrit par `split` : fractions, règles de holdout, raisons
  quality_report.json / .md   écrit par `validate`
```

Convention du projet : **échantillons × canaux** partout (SPEC_V1). Les
fichiers NPZ contiennent toujours `mix` (le signal observé) ; les exemples
synthétiques contiennent en plus `clean` et `noise`, avec `mix = clean + noise`
(puis écrêtage si `saturation_limit` est défini). Aucun tableau n'est normalisé :
l'amplitude physique est conservée.

## Champs du manifeste

| Champ | Type | Obligatoire | Sens |
|---|---|---|---|
| `record_id` | str | oui | identifiant unique dans le jeu |
| `source_id` | str | oui | jeu d'origine (`synthetic_vibration_v1`, `mimii_fan_0db`, …) |
| `specialist` | enum | oui | `vibration`, `acoustic`, `music`, `slow_series` |
| `task_type` | enum | oui | `event_detection`, `event_classification`, `anomaly_detection`, `multilabel_classification` |
| `file_path` | str | oui | chemin relatif au dossier du jeu, séparateur `/` |
| `sample_rate_hz` | float > 0 | oui | fréquence d'échantillonnage réelle du fichier (pas de rééchantillonnage implicite) |
| `n_samples` | int ≥ 4 | oui | doit égaler la première dimension de `mix` |
| `channel_names` | list[str] | oui | uniques |
| `units` | list[str] | oui | même longueur ; `unspecified` pour un WAV ou un signal non calibré, `a.u.` pour le synthétique |
| `labels` | list[str] | — | classes connues (descriptives, voir ci-dessous) ; vide si inconnu |
| `machine_id`, `session_id`, `acquisition_id` | str | — | `unknown` si non disponible ; pour le synthétique : `synthetic`, recette, groupe parent |
| `split` | enum | oui | `train`, `validation`, `test`, `unassigned` |
| `is_synthetic` | bool | oui | |
| `generation` | dict | si synthétique | `components`, `noise`, `saturation_limit`, `seed`, `recipe_id`, `parent_index`, `measured` — suffit à régénérer le fichier à l'identique |
| `provenance` | dict | — | origine, chemin/hash/taille du fichier original, version, DOI, paramètres WAV |
| `license` | str | oui | texte vérifié de la licence source ; `unknown` sinon ; `project-internal-synthetic` pour le synthétique |
| `file_sha256` | str | oui | empreinte du NPZ (vérifiée par `validate`) |
| `snr_db_target` | float ou null | — | SNR demandé (synthétique) |
| `snr_db_measured` | float ou null | — | SNR mesuré sur le fichier (définition ci-dessous) |
| `snr_db_measured_active` | float ou null | — | SNR mesuré sur les zones d'événements seulement |
| `events` | list[{label, start_sample, end_sample}] | — | annotations temporelles `[start, end)` en échantillons, dans `[0, n_samples]` |
| `parent_record_id` | str ou null | — | enregistrement source d'une fenêtre ou d'une augmentation ; hérite du `split` |
| `group_id` | str | oui | unité de séparation train/val/test (machine, session, acquisition ou réalisation parente synthétique) ; `unknown` interdit au split |
| `transforms` | list[dict] | — | opérations traçables appliquées (`window`, `wav_pcm_to_float`, …) |
| `notes` | str | — | |
| `schema_version` | int | oui | 1 |

Métadonnée inconnue = `unknown` (ou `unspecified` pour une unité). On ne
devine jamais une unité physique pour un WAV ni une fréquence d'échantillonnage
pour un CSV : elles sont fournies explicitement à l'import.

## Définition du SNR

```
SNR_dB = 10 · log10( mean(clean²) / mean(noise²) )
```

calculé sur tout l'enregistrement et tous les canaux (`snr_reference = global`,
défaut). Le bruit est mis à l'échelle pour atteindre exactement la cible ; la
valeur réellement obtenue est mesurée et stockée (`snr_db_measured`), puis
recontrôlée par `validate` sur les fichiers float32 (tolérance 0,05 dB).

Pour les événements courts, `snr_db_measured_active` utilise seulement les
échantillons situés dans les zones annotées. Une recette peut demander
`"snr_reference": "active"` dans son bruit : la cible s'applique alors à la zone
active et le SNR global sera bien plus bas (par exemple une impulsion de 40
échantillons à 0 dB actif donne < −15 dB global, testé dans
`tests/test_data_synthetic.py`). Les fenêtres héritent de `snr_db_target` du parent
mais portent leur propre `snr_db_measured` local ; `validate` ne compare la
cible qu'aux enregistrements parents.

## Labels descriptifs, pas de diagnostic

Les labels synthétiques décrivent la forme du signal : `sine`,
`harmonic_series`, `chirp`, `amplitude_modulated`, `isolated_impulses`,
`periodic_impulses`, `irregular_impulses`, `damped_burst`, `noise_burst_modulated`,
`harmonic_hum`, `drift`, `level_change`, `saturated`. Pour l'anomalie :
`reference` vs `deviation` (+ ce qui a été ajouté). Un signal « sinus +
impulsions périodiques » n'est pas un « défaut de roulement confirmé » ; aucun
label mécanique n'est produit par le générateur.

## Aliasing

Toute composante doit respecter `f ≤ 0,9 · fs/2` ; au-delà, le générateur
refuse (`ValueError`) sauf `allow_aliasing: true` explicite dans la composante,
auquel cas les paramètres stockés le montrent. Les harmoniques au-dessus de
cette limite sont supprimées (pas repliées) ; `synthetic.harmonics_kept` donne
le nombre conservé. Aucune recette v1 ne simule l'aliasing.

## Exemple de ligne de manifeste (synthétique, abrégé)

```json
{
  "record_id": "synthetic_vibration_v1__sine_plus_periodic_impulses__p0000__snrp0dB",
  "source_id": "synthetic_vibration_v1",
  "specialist": "vibration",
  "task_type": "multilabel_classification",
  "file_path": "records/synthetic_vibration_v1__sine_plus_periodic_impulses__p0000__snrp0dB.npz",
  "sample_rate_hz": 8192.0, "n_samples": 8192,
  "channel_names": ["synthetic_accel"], "units": ["a.u."],
  "labels": ["sine", "periodic_impulses"],
  "machine_id": "synthetic", "session_id": "sine_plus_periodic_impulses",
  "acquisition_id": "synthetic_vibration_v1/sine_plus_periodic_impulses/p0000",
  "split": "train", "is_synthetic": true,
  "generation": {
    "components": [
      {"type": "sine", "frequency_hz": 201.78, "amplitude": 0.675, "phase_rad": 1.868},
      {"type": "impulses", "kind": "half_sine", "width_samples": 8, "amplitude": 1.130,
       "label": "impulse", "positions_samples": [113, 1347, 2581, "..."]}
    ],
    "noise": {"type": "colored_noise", "exponent": 1.0, "snr_db": 0.0},
    "saturation_limit": null, "seed": 378074516672546249,
    "recipe_id": "sine_plus_periodic_impulses", "parent_index": 0, "generator_version": 1,
    "measured": {"snr_db_measured": 0.0, "snr_db_measured_active": 5.09, "snr_reference": "global",
                 "clean_rms": 0.483, "noise_rms": 0.483, "mix_peak_abs": 2.34, "saturation_fraction": 0.0}
  },
  "provenance": {"origin": "signal_lm.data.synthetic", "config_dataset_id": "synthetic_vibration_v1"},
  "license": "project-internal-synthetic",
  "file_sha256": "07ccce70…",
  "snr_db_target": 0.0, "snr_db_measured": 0.0, "snr_db_measured_active": 5.09,
  "events": [{"label": "impulse", "start_sample": 113, "end_sample": 121}, "..."],
  "parent_record_id": null,
  "group_id": "synthetic_vibration_v1/sine_plus_periodic_impulses/p0000",
  "transforms": [], "schema_version": 1
}
```

## Exemple de ligne pour un WAV importé

```json
{
  "record_id": "mimii_fan_0db__id00__normal_00000000",
  "source_id": "mimii_fan_0db", "specialist": "acoustic", "task_type": "anomaly_detection",
  "file_path": "records/mimii_fan_0db__id00__normal_00000000.npz",
  "sample_rate_hz": 16000.0, "n_samples": 160000,
  "channel_names": ["wav_ch_0"], "units": ["unspecified"],
  "labels": ["normal"], "machine_id": "fan_id00", "session_id": "unknown", "acquisition_id": "unknown",
  "split": "unassigned", "is_synthetic": false,
  "provenance": {"origin": "wav", "original_path": "…/normal_00000000.wav", "original_sha256": "…",
                 "bits_per_sample": 16, "scaling": "int / 2**15", "version": "zenodo 3384388 public 1.0"},
  "license": "CC BY-SA 4.0 (https://zenodo.org/records/3384388)",
  "transforms": [{"op": "wav_pcm_to_float", "scaling": "int / 2**15"}],
  "group_id": "fan_id00", "events": [], "parent_record_id": null, "schema_version": 1
}
```

Le `group_id` d'un enregistrement réel doit être la machine, la session ou
l'acquisition ; c'est la seule information qui empêche les fuites.

## Configuration de génération

Voir `configs/data/synthetic_vibration_v1.json`. Un paramètre peut être :
un nombre (fixe), `[lo, hi]` (uniforme), `{"log_uniform": [lo, hi]}`,
`{"int_uniform": [lo, hi]}`, `{"choice": [...]}`. Par recette, `n_parents`
réalisations sont tirées (graine dérivée de `seed`, `recipe_id`, index), et
chaque parent est rendu une fois par niveau de `snr_db_levels` (`null` = propre).
Toutes les variantes d'un parent partagent `group_id`.

Types de composantes : `sine`, `multi_sine`, `harmonics`, `chirp`,
`amplitude_modulation`, `impulses` (positions explicites, `count`, ou `rate_hz`
+ `jitter_samples`), `damped_oscillation`, `noise_burst`, `drift`, `level_step`.
Bruits : `white_noise`, `colored_noise` (exposant 0/1/2 = blanc/rose/brun),
`band_noise`. `saturation_limit` écrête le mélange.

## Configuration de split

Voir `configs/data/split_v1.json` : `seed`, `fractions`, `stratify_by`
(champ dont chaque valeur est découpée avec les mêmes fractions — `session_id`
= recette pour le synthétique, machine ou source pour le réel), `holdout`
(règles `field` + `range` / `in` / `equals` → `split` forcé, avec `reason`).
