# Passation à Codex — lot « préparation des données » (branche `data-prep`)

## Contexte

Claude Code a livré le lot données sur de **nouveaux chemins** ; aucun fichier
existant n'a été modifié sauf `.gitignore` (`data/` → `/data/`, sinon
`src/signal_lm/data` et `configs/data` étaient ignorés par Git). Le lot vit sur la
branche `data-prep` ; `main` reste à Codex.

Lire d'abord : `docs/DATASET_PLAN.md`, `docs/DATA_SCHEMA.md`, `docs/DATA_SOURCES.md`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\prepare_synthetic_data.ps1      # jeux sous data\, ignorés par Git
```

## Ce qui existe

- Package `src/signal_lm/data/` : `synthetic.py` (générateurs + SNR vérifié),
  `schema.py` (manifeste JSONL), `store.py` (NPZ `[échantillons, canaux]` float32,
  amplitude préservée), `split.py` (split par `group_id` **avant** fenêtrage, holdout,
  `check_leakage`), `windows.py`, `quality.py` (contrôles + `quality_report`),
  `importers.py` (CSV/WAV), `cli.py`.
- Jeux générés et validés (0 erreur) : `data\synthetic_vibration_v1` (672 parents,
  8 192 Hz), `data\synthetic_acoustic_events_v1` (378, 16 kHz),
  `data\synthetic_slow_series_v1` (80, 1 Hz).
- Les labels sont **descriptifs** (`periodic_impulses`, `chirp`, `damped_burst_short`,
  `reference` / `deviation`) : jamais un diagnostic mécanique. Ne pas les renommer en
  « défaut ».

## Mission : brancher ces manifestes au pipeline d'entraînement (jalon 2, SPEC_V1)

Sans toucher aux générateurs ni au schéma.

1. **Lecteur de jeu** : itérer `manifest.jsonl` filtré par `split` / `specialist` /
   `task_type` ; charger via `signal_lm.data.load_signal(root, record)` → `Signal`
   (contrat SPEC_V1) ; empiler en `[batch, samples, channels]` à longueur commune
   (fenêtrer avec `window` avant : le CNN n'a pas de padding masqué).
2. **Cibles** par (`specialist`, `task_type`) : `labels` → index ; `events` → masques
   temporels (`event_detection`) ; `reference`/`deviation` pour `anomaly_detection`
   (entraînement sur `reference` seul) ; multi-label → vecteur binaire.
3. **Normalisation** : statistiques calculées sur `split == "train"` uniquement,
   stockées dans le checkpoint spécialisé, jamais dans les fichiers de données.
4. **Évaluation** : par split, par `snr_db_target` (−10 … 20 dB et propre), par règle de
   holdout (`data\<jeu>\split_summary.json`). Joindre au checkpoint : `dataset_id`,
   `seed` du config, sha256 du manifeste, résultats.
5. Ne jamais lire `split == "test"` pour un choix de seuil ou d'hyperparamètre.

## Contraintes

- Pas de dépendance lourde ; pas de modification de l'architecture sans mise à jour de
  `SPEC_V1` ; `AGENTS.md` s'applique (mesures physiques ≠ interprétation apprise).
- Données réelles : rien n'est importé. Commencer par MIMII fan 0 dB (CC BY-SA 4.0)
  avec `import-wav`, `group_id` = machine ID, split stratifié par `machine_id`
  (voir `DATA_SOURCES.md` pour les licences vérifiées).
