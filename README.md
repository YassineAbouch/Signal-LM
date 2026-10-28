# Signal-LM

Petit modèle de perception de signaux **initialisé de zéro**, destiné à devenir
un spécialiste après entraînement. Python / PyTorch ; Windows comme poste cible.

## État du projet

Lot 1 : modèle vierge configurable, sauvegarde/rechargement, arrays et CSV,
FFT/RMS et tests synthétiques. **Aucun spécialiste entraîné pour l'instant.**
Les mesures fonctionnent immédiatement. Les embeddings et logits du réseau
vierge n'ont pas de signification métier. Le pipeline d'apprentissage complet,
le streaming, les spectrogrammes et la détection d'anomalies viennent ensuite.

## Installation Windows (PowerShell)

Si le dossier local n'est pas encore un clone, depuis son dossier parent :

```powershell
git clone https://github.com/YassineAbouch/Signal-LM.git E:\Signal-LM
Set-Location E:\Signal-LM
```

Si le dépôt est déjà cloné, ouvrir ce dossier et inspecter `git status` avant de
faire `git pull --ff-only`. Ne pas supprimer un dossier existant pour recloner.

Créer un environnement Python compatible avec la version PyTorch choisie.
Sur le poste Windows validé, Python 3.13.2 est installé :

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Pour la carte NVIDIA, relever le modèle/pilote avec `nvidia-smi`, puis choisir
Windows / Pip / Python et la plateforme CUDA compatible dans le
[sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/).
Exécuter sa commande avec `.\.venv\Scripts\python.exe -m pip` à la place de
`pip` ou `pip3`. Ne pas choisir une version CUDA au hasard.
Pour vérifier le socle sur CPU uniquement, alternative :

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Puis installer le projet et vérifier :

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m signal_lm.cli doctor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m signal_lm.cli init --output artifacts/virgin.pt
.\.venv\Scripts\python.exe -m signal_lm.cli demo
.\.venv\Scripts\python.exe scripts/validate_runtime.py
```

`init` refuse d'écraser un checkpoint existant. `demo` ne charge ni n'entraîne le
réseau : elle vérifie les mesures sur une sinusoïde de 120 Hz.
Un `cuda_available: false` n'empêche pas les tests CPU, mais signifie que
l'accélération CUDA n'est pas disponible dans cet environnement.
Le script `validate_runtime.py` vérifie le checkpoint par défaut, son refus
d'écrasement, puis le calcul et les gradients CUDA si disponibles, sans
optimiseur ni mise à jour des poids. Il conserve un checkpoint déjà présent et
échoue si celui-ci ne correspond pas à l'initialisation par défaut attendue.

Pour reproduire l'environnement Windows validé, installer d'abord PyTorch :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-cu130.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Ces versions décrivent ce poste Python 3.13 / NVIDIA ; elles ne remplacent pas
le choix d'un paquet compatible pour un autre matériel.

## Interface web locale

Depuis `E:\Signal-LM`, lancer :

```powershell
.\.venv\Scripts\python.exe -m signal_lm.web
```

Ouvrir http://127.0.0.1:8765 dans le navigateur. Arrêter avec `Ctrl+C`.
Le port peut être choisi avec `--port 8766`.

L'interface propose une sinusoïde configurable, l'import CSV avec sélection
explicite des colonnes et unités, le choix du canal, les courbes temporelle/FFT,
les mesures physiques, l'export JSON et le diagnostic PyTorch/CUDA.
La page `/model` présente l'architecture et la configuration du modèle par
défaut, la taille des poids et l'environnement CPU/CUDA. Le diagnostic peut
être actualisé ; il inspecte une instance fraîche sans charger de checkpoint.
Les valeurs CSV sont traitées en mémoire par le serveur local, sans sauvegarde
automatique. L'export JSON est un téléchargement volontaire contenant les mesures
et les valeurs affichées ; il est à conserver hors Git pour des données réelles.
La démo démarre automatiquement ; aucun fichier réel n'est chargé par défaut.

Limites : requête de 2 Mio, 100 000 échantillons, 8 canaux. La courbe temporelle
est un aperçu réduit à 1 800 points au maximum, pouvant masquer des transitoires ;
les mesures et la FFT utilisent tous les échantillons. La FFT affichée conserve
les maxima des groupes de points si le spectre est réduit pour l'affichage.
Les timestamps CSV ne sont pas contrôlés. Aucun diagnostic appris ni entraînement.
Le serveur écoute uniquement sur `127.0.0.1` : cette version n'est pas un service
Internet ou multi-utilisateur. Aucun nouvel outil JavaScript n'est à installer.

## Analyse d'un array

```python
import numpy as np
from signal_lm import Signal, analyze

fs = 4096
t = np.arange(fs) / fs
signal = Signal(np.sin(2*np.pi*120*t), fs, ("acceleration_x",), ("g",))
result = analyze(signal, include_spectrum=True)
print(result["measurements"])
```

## Analyse CSV (lorsque les données seront disponibles)

```powershell
.\.venv\Scripts\python.exe -m signal_lm.cli analyze-csv data/signal.csv --sample-rate 16000 --columns acceleration_x --units g
```

En-tête obligatoire ; utiliser `--delimiter ";"` si nécessaire. Les colonnes temps
ne doivent pas être sélectionnées comme mesures. La V1 suppose un échantillonnage
régulier ; elle ne vérifie pas les timestamps du CSV.

## Utilisation du modèle vierge

```python
import torch
from signal_lm.model import load_virgin

model = load_virgin("artifacts/virgin.pt")
with torch.no_grad():
    result = model(torch.zeros(1, 4096, 1))  # batch, samples, channels
print(result["embedding"].shape)  # [1, 64], aucune interprétation apprise
```

## Collaboration

La spécification est dans [docs/SPEC_V1.md](docs/SPEC_V1.md).
Le cadrage du premier spécialiste et son protocole proposé sont dans
[docs/SPECIALIZATION_PLAN.md](docs/SPECIALIZATION_PLAN.md) ; les décisions métier
restent à définir avant entraînement.
Codex et Claude Code suivent [AGENTS.md](AGENTS.md) ; `CLAUDE.md` y renvoie.
Travailler séquentiellement ou sur des branches distinctes.
Les données et checkpoints sont exclus de Git. Les dépendances autorisent une
plage de versions ; consulter `docs/VALIDATION.md` pour l'environnement testé.
