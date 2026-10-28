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

## Exécution et résultats

Exécution à partir d'une copie figée du commit `5b80e74` sur `data-prep`.
Les modifications fusion non commitées du dossier de travail n'ont pas été
utilisées. Suite complète de cette copie : **104 tests réussis en 16,67 s**.
Windows 11, Python 3.13.2, PyTorch 2.13.0+cu130, NumPy 2.5.4,
NVIDIA RTX 4070 Laptop GPU.

Le pilote source reste inchangé. Le jeu dérivé contient 200 parents et
2 000 fenêtres non recouvrantes, avec **0 erreur et 0 avertissement** qualité.
L'apprentissage et la normalisation utilisent 500 fenêtres issues des
50 clips normaux train. La sélection et le seuil utilisent les références
validation (250 fenêtres, 25 clips), sans lecture des tableaux test dans `fit`.

Les 10 époques et 320 étapes d'optimisation sont terminées. L'époque 10 a été
sélectionnée par perte validation minimale sur références (0,0000214585).
Durée complète préparation + apprentissage + évaluation : environ 188 s.
Les baisses de perte ne sont pas une mesure de compétence métier.

### Résultats par clip de 10 secondes

Chaque partition validation/test contient 25 clips normaux et 25 anormaux.
Les scores et seuils par clip suivent la règle fixée avant le test.

| Partition / méthode | Anormaux repérés | Fausses alertes sur normaux | F1 anomalie | AUROC |
| --- | ---: | ---: | ---: | ---: |
| Validation, réseau | 25 / 25 | 2 / 25 | 0,9615 | 1,0000 |
| Validation, RMS | 25 / 25 | 2 / 25 | 0,9615 | 1,0000 |
| Test, réseau | 17 / 25 | 25 / 25 | 0,5075 | 0,1616 |
| Test, RMS | 0 / 25 | 25 / 25 | 0,0000 | 0,0000 |

Le réseau a donc un **taux de faux positifs test de 100 %**, avec 8 anomalies
manquées et une exactitude globale de 34 %. Le résultat n'est pas acceptable
pour une utilisation fiable. La bonne séparation en validation ne se retrouve
pas sur la machine test réservée. Le RMS échoue également.

Au niveau fenêtre (500 fenêtres test), le réseau obtient un rappel de 45,6 %,
85,6 % de faux positifs et une AUROC de 0,2220. Les 500 fenêtres ne doivent pas
être traitées comme 500 machines indépendantes ; il n'y a qu'une machine test.

Observation descriptive, après test : le RMS médian des fenêtres normales est
0,00755 en validation et 0,00908 en test ; celui des fenêtres anormales est
respectivement 0,00863 et 0,00554. Les niveaux et leur relation aux labels
diffèrent entre ces deux machines. Cela suggère un décalage de distribution,
sans démontrer à lui seul la cause complète de l'échec du réseau. Le seuil
n'a pas été retouché et le sens du score n'a pas été inversé à partir du test.

### Checkpoints et traçabilité

Dossier local, ignoré par Git : `artifacts/experiments/mimii_fan_real_v1/`.

- `trained.pt` : poids sélectionnés et résultats train/validation.
- `frozen_before_test.json` : empreinte des poids et seuils avant accès test.
- `final.pt` : même modèle, enrichi des évaluations test et de la comparaison RMS.
- `protocol.json`, `progress.jsonl`, `validation_scores.json`, `test_scores.json`,
  `quality_summary.json`, `result.json` : protocole, historique et résultats détaillés.

SHA-256 de `final.pt` :
`e2ed2d746cabeed3b66bf5b77bf62e0fe22b04e1023182bd0c3efd59f7d94898`.
Relecture `weights_only=True` et sorties CPU strictement identiques vérifiées.
Poids, normalisation et seuil sont identiques avant et après l'évaluation finale.
Le checkpoint vierge existant est conservé ; aucun poids ou fichier réel n'est
ajouté à Git. Le statut spécialisé est bien `trained`, mais ne signifie pas validé.

## Suite recommandée

Examiner les variations entre machines sur train/validation et comparer des
représentations moins sensibles au niveau sonore, avec un protocole d'évaluation
indépendant fixé avant les prochains réglages. Le résultat de la machine 06 est
désormais connu : ne pas utiliser ce même test pour choisir une transformation,
un seuil ou une architecture puis le présenter comme un nouveau test aveugle.
Ce premier modèle reste un résultat expérimental ; il n'est pas activé dans
l'interface audio.
