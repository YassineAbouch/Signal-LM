# Sources de données réelles — état vérifié au 2026-10-28

Rien n'a été téléchargé. Chaque ligne ci-dessous a été vérifiée sur la page
officielle (ou la plus proche accessible) à la date indiquée ; « non indiqué »
signifie que la page ne le dit pas, pas que c'est libre. **Un jeu accessible
publiquement n'est pas forcément réutilisable ni redistribuable** ; tout
usage commercial exige de relire la licence et, pour les licences NC, de
contacter les auteurs.

Légende usage prévu : `D` détection d'événement, `C` classification de
classe connue, `A` anomalie vs référence, `M` multi-label.

## Vibrations industrielles

| Source | Lien officiel / version | Licence vérifiée | Capteurs, Fs, unités | Classes / conditions | Volume, format | Limites | Usage Signal-LM |
|---|---|---|---|---|---|---|---|
| CWRU Bearing Data Center | https://engineering.case.edu/bearingdatacenter (pas de numéro de version) | **Aucune licence ni condition d'utilisation indiquée** sur la page (© CWRU). À clarifier avant tout usage hors recherche interne. | Accéléromètres DE/FE ; 12 kHz et 48 kHz (d'après la littérature, la page d'accueil ne le dit pas) ; unités non indiquées sur la page | Sain + défauts EDM bague interne / bille / bague externe, 0,007–0,040 in, 0–3 hp, 1720–1797 rpm | quelques centaines de .mat | Banc de labo, un seul moteur, défauts artificiels EDM, faible diversité, fuites fréquentes dans la littérature (fenêtres voisines dans train et test) | C, A (benchmark de référence, pas de preuve terrain) |
| Paderborn (KAt-DataCenter) | https://mb.uni-paderborn.de/en/kat/research/kat-datacenter/bearing-datacenter | **CC BY-NC 4.0** ; usage académique non commercial avec citation (Lessmeier et al., PHM 2016) ; « pour usage commercial contacter l'auteur » | Vibration + courant moteur ; Fs non indiquée sur la page (64 kHz d'après la littérature) | Défauts artificiels et réels (fatigue), 32 roulements d'après la littérature | .mat, plusieurs Go | **Non commercial** ; banc de labo | C, A — recherche uniquement tant que l'accord commercial n'existe pas |
| NASA IMS Bearings (Univ. Cincinnati) | https://data.nasa.gov/dataset/ims-bearings — fichier `IMS.zip` | Champ licence = « other-license-specified » pointant vers https://www.usa.gov/government-works ; pas de texte de licence propre au jeu | Vibration ; Fs, canaux non indiqués sur la page (20 kHz, 4 roulements d'après la littérature) | 3 essais run-to-failure, pas de labels de classe, seulement la chronologie jusqu'à la panne | ASCII, ~1 Go | Pas de labels d'état ; annotation à construire (fenêtres « début de vie » vs « fin de vie ») ; 3 essais seulement | A (référence = début de vie), pas de C |
| MAFAULDA (UFRJ) | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html | **Page inaccessible depuis cet environnement** (robots) ; licence non vérifiée | 8 canaux : tachymètre, 2×3 axes accéléro, micro ; Fs annoncée 50 kHz par la littérature | sain, balourd, désalignement, défauts roulements | CSV, ~13 Go | Licence à vérifier manuellement ; banc de labo | C, M (vibration + audio synchrones) — après vérification |
| SKAB (Skoltech) | https://github.com/waico/SKAB — DOI 10.34740/KAGGLE/DSV/1693952 | **GPL-3.0** (dépôt) ; la licence des données elles-mêmes suit le dépôt, à confirmer | RMS accéléro ×2 (g), courant (A), pression (bar), température (°C) ×2, tension (V), débit (L/min) ; période d'échantillonnage non indiquée (≈1 s) | 35 CSV, colonnes `anomaly` et `changepoint` | petit (Mo) | Signaux RMS déjà agrégés, pas de forme d'onde ; banc de labo ; GPL = contrainte de diffusion si redistribué | A (séries lentes + changepoints) |

Candidats non vérifiés dans cette session (liens de la liste `awesome-bearing-dataset`) : XJTU-SY, FEMTO/PRONOSTIA, MFPT, JNU, HUST, Ottawa, SEU, PHM09 gearbox. Licences à vérifier une par une avant tout usage.

## Bruits de machines et événements acoustiques

| Source | Lien officiel / version | Licence vérifiée | Capteurs, Fs | Classes / conditions | Volume, format | Limites | Usage |
|---|---|---|---|---|---|---|---|
| MIMII | https://zenodo.org/records/3384388 — DOI 10.5281/zenodo.3384388, « public 1.0 » (2019) | **CC BY-SA 4.0** (attribution + partage à l'identique des dérivés) | Réseau de 8 micros ; 16 kHz, 16 bits ; unité : non calibrée (`unspecified`) | valve, pump, fan, slider ; 4 modèles par type ; normal 5 000–10 000 s / anormal ≈1 000 s par modèle ; anomalies : contamination, fuite, balourd, rail abîmé ; bruit d'usine mêlé à −6/0/+6 dB | 100 Go en 12 zips (fan −6 dB : 10,9 Go) | Volumineux ; SA impose de partager les dérivés sous la même licence ; sous-ensemble conseillé (un type machine, un SNR) | A, C, D |
| DCASE 2020 Task 2 dev | https://zenodo.org/records/3678171 — DOI 10.5281/zenodo.3678171 | **CC BY-NC-SA 4.0** — **non commercial** | Mono 16 kHz WAV, 10 s (1er canal de MIMII/ToyADMOS) | ToyCar, ToyConveyor, valve, pump, fan, slider ; ~1 000 clips normaux d'entraînement par ID, 100–200 normaux/anormaux en test | 8,1 Go (6 zips de 1,0–1,9 Go) | NC : recherche interne uniquement ; déjà rééchantillonné | A (benchmark de référence) |
| ToyADMOS | https://zenodo.org/record/3351307 | Licence dans `LICENSE.pdf` (non lue ici) — **à vérifier** | 4 micros, 48 kHz | toy car / conveyor / train ; ~540 h normal, >12 000 anomalies | ~197 Go (7z) | Trop gros pour un premier lot ; licence à lire | A |
| ToyADMOS2 | https://zenodo.org/record/4580270 (arXiv 2106.02369) | Licence dans `LICENSE.pdf` (non lue ici) — **à vérifier** | 48 kHz, MPEG-4 ALS (.mp4 → décodage requis) | toy car / toy train avec décalages de domaine ; >27k normal / >8k anormal par sous-jeu ; jeux d'exemple 13,2 Go | 169 Go | Format ALS non lisible par `wave` ; licence à lire | A (domain shift) |
| ESC-50 | https://github.com/karolpiczak/ESC-50 | **CC BY-NC** (clips) ; sous-ensemble ESC-10 en CC BY | Mono 44,1 kHz WAV, 5 s | 50 classes × 40 clips (dont `door knock`, `engine`, etc.), 5 folds | ~600 Mo | NC ; classes d'ambiance générale, peu de bruits machine ; clips de qualité hétérogène | C (événements simples : chocs, frappes) — recherche uniquement |
| FSD50K | https://zenodo.org/records/4060432 — DOI 10.5281/zenodo.4060432 | Jeu en CC BY, **licence par clip** (CC0 / CC BY / CC BY-NC / Sampling+) dans `dev_clips_info_FSD50K.json` ; filtrer les clips NC pour tout usage commercial | Mono 44,1 kHz 16 bits WAV, 0,3–30 s | 200 classes (ontologie AudioSet), labels faibles au niveau clip ; dev 40 966 clips, eval 10 231 | 24,7 Go | Labels faibles (pas de début/fin) ; filtrage par licence indispensable | D (après filtrage), C |

## Musique — identification d'instruments

| Source | Lien / version | Licence vérifiée | Fs, format | Classes | Volume | Limites | Usage |
|---|---|---|---|---|---|---|---|
| NSynth | https://magenta.tensorflow.org/datasets/nsynth | **CC BY 4.0** | 16 kHz, WAV 16 bits (JSON/WAV) ou TFRecord ; notes de 4 s | 305 979 notes, 11 familles d'instruments, un instrument par note | plusieurs dizaines de Go (train) ; `nsynth-test` et `nsynth-valid` sont les sous-ensembles à prendre d'abord | Notes isolées : pas de polyphonie, pas de mélange | C (mono-instrument) |
| OpenMIC-2018 | https://zenodo.org/records/1432913 — DOI 10.5281/zenodo.1432913, v1.0.0 | **CC BY 4.0** pour le jeu ; **licence par piste audio** dans `openmic-2018-metadata.csv` (audio issu de la Free Music Archive) | OGG, 10 s ; Fs non indiquée sur la page (44,1 kHz d'après le papier ISMIR 2018) | 20 instruments, annotations partielles (présent / absent / inconnu) → multi-label avec labels manquants | 2,6 Go | OGG à décoder (pas de lecteur dans le socle) ; labels partiels | M |

## Séries industrielles lentes

| Source | Lien / version | Licence vérifiée | Variables, période | Volume | Limites | Usage |
|---|---|---|---|---|---|---|
| SKAB | voir ci-dessus | GPL-3.0 (dépôt) | 8 variables physiques, ≈1 s, `anomaly`/`changepoint` | Mo | GPL ; banc de labo | A, D (changepoints) |
| UCI Air Quality | https://archive.ics.uci.edu/dataset/360/air+quality — DOI 10.24432/C59K5F | **CC BY 4.0** | Moyennes horaires mars 2004–fév. 2005 ; 5 capteurs MOX + T, RH, AH + référence ; 9 358 lignes ; manquants codés **−200** | 0,8 Mo CSV | Pas de labels d'anomalie ; manquants à gérer explicitement (jamais imputés en silence) | dérives, saisonnalité (A non supervisé) |
| NASA IMS | voir ci-dessus | government-works (champ licence) | run-to-failure | ~1 Go | pas de labels | A |

## Premier sous-ensemble recommandé (petit, droits clairs)

1. **MIMII `fan`, SNR 0 dB** (10,4 Go, CC BY-SA 4.0) — ou une seule machine ID si le poste est limité ; normal vs anormal, 16 kHz, WAV lisible par `import-wav`.
2. **NSynth `nsynth-test`** (CC BY 4.0, quelques milliers de notes) pour la couche musique.
3. **UCI Air Quality** (CC BY 4.0, 0,8 Mo) pour les séries lentes.
4. CWRU seulement après clarification écrite de ses conditions ; Paderborn / DCASE / ESC-50 en recherche interne uniquement (NC).

Rien de tout cela n'est importé pour l'instant : le pipeline a été vérifié sur
le jeu synthétique uniquement.
