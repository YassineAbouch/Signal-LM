# Premier entraînement réel — MIMII fan 0 dB

Autorisation utilisateur du 28 octobre 2026 : « lance le entraînement reel ».
Le protocole est fixé avant l'entraînement et la lecture des résultats test.
Il s'agit d'un spécialiste acoustique de détection d'écart à une référence,
pas d'un modèle de langage ou d'un diagnostic de défaut mécanique.

## Protocole préfixé

- Pilote MIMII de 200 clips : 25 normaux / 25 anormaux par machine, 4 machines.
- Train : machines 00/02, **références normales seules** pour statistiques,
  centre d'embedding et optimisation. Validation : machine 04. Test : machine 06.
- Copie indépendante du jeu et fenêtres de 16 000 échantillons (1 s) à 16 kHz,
  8 canaux, pas de 16 000, sans recouvrement ni changement d'amplitude.
- CNN existant sans tête de classes, objectif une-classe déjà spécifié dans
  `SPEC_V1.md`. Aucun poids externe ni modification d'architecture.
- 10 époques, batch 16, seed 42, AdamW, lr 0,001, weight decay 0,0001, CUDA.
- Sélection : minimum de perte sur références validation ; seuil d'anomalie
  au 95e percentile des scores des références validation.
- Score par clip de 10 s : moyenne des scores de ses fenêtres. Seuil par clip
  également fixé sur les références validation seules.
- Référence simple : RMS physique global sur les 8 canaux, avant normalisation ;
  score par clip = moyenne des RMS de ses fenêtres. Même quantile validation.
- Les seuils et l'empreinte du checkpoint sont écrits avant évaluation test.
  Aucun réglage n'est effectué à partir du test.

Les tableaux train/validation peuvent être mis en cache immuable en mémoire.
Leur intégrité est vérifiée au chargement et avant sauvegarde. Les données
sources ne reçoivent aucune normalisation ou écriture.

## Commande reproductible

Depuis une copie du code commité, avec des chemins de sortie encore inexistants :

```powershell
.\.venv\Scripts\python.exe -m signal_lm.training.real_run `
  --source data/mimii_fan_0db `
  --root data/mimii_fan_0db_windows_real_v1 `
  --run-dir artifacts/experiments/mimii_fan_real_v1
```

`protocol.json`, `progress.jsonl`, `trained.pt`, `frozen_before_test.json`,
les scores validation/test, `final.pt` et `result.json` sont enregistrés sous
le dossier d'expérience ignoré par Git. Les poids vierges ne sont pas remplacés.
Le checkpoint final conserve exactement les poids, normalisation et seuil
du checkpoint sélectionné. La relecture est vérifiée sur CPU.

## Limites d'interprétation

Le pilote ne comporte que deux machines train, une validation et une test.
Les fenêtres d'un même clip ou d'une même machine sont corrélées. Les métriques
principales sont donc aussi rapportées par clip, avec effectifs explicites.
Aucun intervalle de généralisation à tout parc industriel n'est revendiqué.
Le bruit est celui du sous-ensemble 0 dB. La sélection initiale par noms de
fichiers peut biaiser le pilote. Un réseau entraîné peut rester peu utile :
la comparaison au RMS et le taux de faux positifs doivent être rapportés.

L'interface audio existante n'active pas automatiquement ce spécialiste : elle
reste dédiée aux mesures physiques, sans confiance métier calibrée.
