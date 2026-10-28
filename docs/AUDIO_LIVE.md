# Audio du PC en temps réel

## Lancement Windows

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[audio]"
.\.venv\Scripts\python.exe -m signal_lm.web
```

Ouvrir http://127.0.0.1:8765/audio. Choisir la sortie utilisée par l'application
audio, renseigner la zone et cliquer **Démarrer la capture**. Lancer un son sur
cette sortie. **Arrêter la capture** ferme le périphérique et libère le tampon.
Le microphone n'est pas utilisé et rien n'est enregistré sur disque.

## Analyser un fichier WAV ou la piste sonore d'un AVI

Dans `/audio`, choisir **Fichier audio**, puis **Choisir un fichier audio / AVI**.
Renseigner la zone et le début de fenêtre en secondes, puis **Analyser le fichier**.
Changer la position et relancer pour explorer une autre fenêtre.

FFmpeg et ffprobe doivent être dans PATH (versions 7.1.1 présentes sur ce poste).
Le serveur reçoit le fichier localement en mémoire, extrait la première piste
audio et analyse au plus une seconde. WAV, MP3, FLAC et OGG ainsi que les pistes
audio des conteneurs AVI sont pris en charge selon les codecs FFmpeg disponibles.
Un AVI sans piste sonore produit une erreur explicite. Les fichiers qui demandent
un accès disque aléatoire et ne se décodent pas par pipe peuvent être refusés.

Limites : 50 Mio par fichier, 192 kHz, 8 canaux, délai de décodage de 30 secondes
par opération. Le fichier n'est pas écrit sur disque, la fréquence et les canaux
sont conservés, sans normalisation d'amplitude. L'amplitude décodée est exprimée
en FS. Une seule analyse de fichier est exécutée à la fois. Aucun son n'est joué.

La fenêtre se termine avant une seconde si le fichier se termine plus tôt.
L'horodatage correspond à l'analyse ; la date d'acquisition reste inconnue et
la position dans le fichier est exprimée séparément en secondes. Changer de
source arrête d'abord une éventuelle capture live de cet onglet.

Validation : suite complète de 52 tests réussis ; WAV synthétique stéréo,
amplitude/RMS et position de fenêtre, extraction d'un AVI contenant de l'audio,
fichier invalide, position hors fichier et absence de FFmpeg. Dans le navigateur,
`qa-tone.wav` synthétique à 48 kHz est chargé et analysé : 440 Hz, RMS 0,1768 FS,
pic 0,25 FS. Aucun fichier réel utilisé pour ces tests.

Le backend utilise PyAudioWPatch 0.2.12.9 / WASAPI loopback sur Windows :
https://github.com/s0d3s/PyAudioWPatch. L'extension `audio` est facultative ;
le reste de Signal-LM fonctionne sans ce module.

## Contrat du flux

- Fréquence native de la sortie sélectionnée, sans rééchantillonnage implicite.
- Canaux conservés séparément, maximum 8 ; maximum 192 kHz.
- Échantillons float32 convertis en `Signal` pour les mesures existantes.
- Unité `FS` : amplitude numérique référencée à la pleine échelle, sans
  conversion en pression acoustique, accélération ou dB SPL.
- Fenêtre glissante d'une seconde. L'interface demande la dernière fenêtre
  environ toutes les 250 ms après la réponse précédente ; ce n'est pas une
  garantie de cadence ou de latence temps réel stricte.
- Courbe temporelle réduite à 1 800 points ; mesures et FFT sur la fenêtre complète.
- Les callbacks restent dans un tampon borné à environ une seconde plus un bloc.
  Une rupture signalée par WASAPI vide le tampon avant de reconstruire une
  fenêtre ; le compteur de ruptures est affiché.
- Le silence sans callbacks, la déconnexion ou l'absence de nouvelles données
  mettent l'affichage en attente. Les anciennes courbes restent figées et
  portent leur ancien horodatage ; aucun zéro artificiel n'est ajouté.

## Zone et date

Chaque fenêtre contient `capture.zone`, `source=wasapi_loopback`, le nom du
périphérique, un numéro de séquence et les bornes `window_start_utc` /
`window_end_utc`. L'heure locale est affichée dans le navigateur.
La fin de fenêtre est estimée à l'heure de réception du callback sur le PC ;
le début est déduit de la durée des échantillons. Il ne s'agit pas d'une
synchronisation métrologique avec d'autres machines ou capteurs.

La zone est une étiquette renseignée par l'utilisateur et reste fixe pendant
une capture. Arrêter puis redémarrer pour changer de zone ou de sortie.
Le tampon n'est pas un historique consultable par date : il est remplacé au fil
du flux. Un futur stockage nécessitera une demande explicite et un contrat
d'acquisition distinct. Aucun jeu d'entraînement n'est constitué automatiquement.

## Cycle de vie et accès

Une seule capture est active par serveur. Chaque démarrage retourne un jeton
de session uniquement conservé en mémoire dans l'onglet ; un autre onglet ne
peut pas arrêter cette capture sans ce jeton. Les requêtes de suivi entretiennent
une durée de vie de 15 secondes. Sans suivi, le périphérique est fermé et le
tampon effacé. La navigation hors de la page tente aussi un arrêt immédiat.
Une mise en veille du navigateur peut déclencher cet arrêt : redémarrer ensuite.

Le serveur reste sur `127.0.0.1`, refuse les hôtes non locaux et les requêtes
POST d'origine étrangère. Ce serveur n'est pas un service Internet authentifié.
La capture expose tout son diffusé par la sortie choisie pendant son activation.

## Validation du 28 octobre 2026

- PyAudioWPatch 0.2.12.9 installé sous Python 3.13.2.
- Trois sorties WASAPI détectées : NVIDIA, Oculus et Realtek, 48 kHz stéréo.
- Navigateur : chargement des sorties, démarrage, attente de données et arrêt
  confirmés. Aucun fichier audio enregistré. Aucune fenêtre complète reçue
  pendant ce contrôle ; tracé de son réel et latence matérielle non validés.
- 45 tests du périmètre exécutés avec
  `python -m pytest -q tests/test_model.py tests/test_signals.py tests/test_web.py tests/test_audio.py`.
  Vérifications audio : deux canaux synthétiques, fréquence/RMS, unités,
  zone/horodatages UTC, tampon borné, ruptures, isolation des sessions,
  arrêt et expiration après absence de suivi. Le périphérique des tests est simulé.
- JavaScript vérifié avec `node --check`.
- D'autres fichiers de préparation des données présents dans l'espace de
  travail sont indépendants de cette fonctionnalité et n'ont pas été modifiés.

Le modèle reste vierge : aucune classification métier, calibration de confiance
ou compétence acoustique apprise n'est revendiquée.
