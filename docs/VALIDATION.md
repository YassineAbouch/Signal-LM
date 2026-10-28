# Validation du lot 1

Vérification dans l'environnement de développement Linux CPU :

- Python 3.12.14 ; PyTorch 2.15.0+cpu ; NumPy 2.3.5 ; pytest 9.2.0.
- Installation du package en mode éditable réussie.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q` : 22 tests réussis.
- CLI `doctor`, `init --output artifacts/virgin.pt` et `demo` exécutées.
- Configuration par défaut : 22 304 paramètres, 89 216 octets de poids float32
  (environ 87 Kio), hors sérialisation et mémoire d'activations.
- Démo : fréquence dominante 120 Hz, amplitude 1, RMS 0.7071067811865476.
- Correction testée : recherche du pic avant mise à l'échelle unilatérale,
  afin d'éviter de déplacer un pic de Nyquist vers le lobe voisin de Hann.

La sauvegarde/relecture reproduit exactement les sorties, la graine est
reproductible, les gradients sont finis et les entrées invalides sont rejetées.
Ces tests valident le socle logiciel, pas une compétence d'analyse industrielle.

Non vérifiés dans cette session : exécution Windows, pilote NVIDIA/CUDA du poste,
latence et RAM sur matériel cible, entraînement et précision métier.
Les dépendances du projet sont déclarées par plages ; les versions ci-dessus
décrivent cette exécution, pas une exigence de version CUDA pour le poste Windows.

## Validation Windows / NVIDIA — 28 octobre 2026

- Import de l'archive v0.1.0 dans `E:\Signal-LM`, initialement vide, sans écrasement.
- Windows 11 (build 26200), Python 3.13.2, environnement `.venv` isolé.
- NVIDIA GeForce RTX 4070 Laptop GPU, 8 188 Mio, pilote 610.47.
- PyTorch 2.13.0+cu130, runtime CUDA 13.0 ; NumPy 2.5.4 ; pytest 9.2.0.
- Installation : `python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130`,
  puis `python -m pip install -e ".[dev]"`, avec le Python de `.venv`.
- Source des paquets CUDA Windows : https://pytorch.org/get-started/previous-versions/.
- `python -m signal_lm.cli doctor` : `cuda_available: true`, GPU correctement reconnu.
- `python -m pytest -q` : **22 passed in 6.78s**.
- `python -m signal_lm.cli demo` : 120 Hz, amplitude 1.0000000000000002,
  RMS 0.7071067811865476 ; prédiction indisponible, aucun spécialiste entraîné.
- `python scripts/validate_runtime.py` : création de `artifacts/virgin.pt`,
  22 304 paramètres, `status=untrained`, `training_steps=0` ; sorties CPU
  strictement identiques après relecture, écrasement refusé et fichier inchangé.
- Sur CUDA : forward et backward finis, sorties proches du CPU
  (`rtol=1e-4`, `atol=1e-5`), paramètres inchangés ; aucun optimiseur exécuté.
- CLI `init --output artifacts/cli-virgin.pt` réussie ; `python -m pip check` :
  aucune dépendance incompatible. Versions figées dans `requirements-windows-cu130.txt`.

Le checkpoint et `.venv` restent exclus de Git. La suite pytest utilise le CPU ;
le script complémentaire vérifie le modèle par défaut sur GPU. Aucun entraînement
métier, aucune donnée réelle, mesure de latence ou précision industrielle.

## Interface web locale — 28 octobre 2026

- Serveur Python standard sur `127.0.0.1`, interface HTML/CSS/JavaScript sans CDN.
- Suite complète : 30 tests réussis, dont démo web, sélection CSV/unités,
  entrées invalides, routes HTTP, refus d'origine étrangère et chemin inconnu.
- JavaScript vérifié par `node --check` ; roue Python construite avec les assets web.
- Navigateur : démo initiale 120 Hz, modification à 50 Hz et actualisation des
  mesures/courbes vérifiées ; GPU CUDA et nombre de paramètres affichés.
- Aucun fichier réel utilisé. Import fichier et export JSON non vérifiés par
  interaction navigateur ; parsing CSV et endpoint couverts par les tests.
- Mise en page adaptative prévue ; rendu mobile non vérifié dans cette session.
- Aucun hébergement Internet, contrôle d'accès multi-utilisateur ou entraînement.

### Correction de la navigation Modèle

- Page `/model` dédiée et lien de navigation actif ; ancien lien `/#model`
  redirigé vers cette page.
- Architecture, configuration, paramètres, taille des poids et matériel issus
  du diagnostic du serveur, avec actualisation et message d'erreur explicite.
- Les tests HTTP vérifient la page, son JavaScript et les métadonnées du modèle
  par défaut ; suite complète : 30 tests réussis en 5,95 s.
- Navigateur : ancienne URL, affichage des données CUDA, bouton d'actualisation
  et retour à l'analyse vérifiés. Aucun checkpoint modifié ou entraînement lancé.
