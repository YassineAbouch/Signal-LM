# Mode Learning : annotations audio

Ouvrir http://127.0.0.1:8765/learn après `python -m signal_lm.web`.

1. Choisir un fichier WAV, M4A, MP3 ou autre format audio pris en charge (50 Mio maximum).
2. Indiquer la zone et la position : une fenêtre d'au plus une seconde est décodée en mémoire.
3. Glisser sur la courbe, ou saisir les bornes relatives dans les champs numériques.
4. Donner un libellé descriptif et une signification, puis éventuellement un identifiant machine stable.
5. Cliquer sur **Enregistrer pour apprentissage**. Chaque clic valide conserve uniquement le segment et ses métadonnées sous `data/manual_audio_annotations_v1/`, ignoré par Git.

Les NPZ conservent tous les canaux, les amplitudes numériques et la fréquence d'origine. Les indices sont arrondis aux échantillons ; au moins quatre points sont requis. L'aperçu réduit à 1 800 points n'est pas le signal sauvegardé. Le fichier complet n'est pas copié ; son nom et son SHA256 servent à tracer l'acquisition. La date d'acquisition reste inconnue, distincte de la date d'annotation. Un M4A est décodé en PCM : cela ne restitue pas une qualité perdue à la compression.

Le manifeste utilise le schéma existant : spécialiste `acoustic`, tâche `event_classification`, libellé du segment, provenance contenant la description utilisateur. Ce n'est pas une supervision texte d'un LLM. Les annotations restent `unassigned` et `unverified_user_annotation` ; aucun entraînement n'est déclenché et aucune annotation n'est présentée comme un diagnostic vérifié. Une soumission strictement identique est idempotente. Le fichier entier est regroupé par SHA256, ou par identifiant machine si renseigné ; changer le groupe d'un fichier déjà annoté est refusé.

Avant un entraînement ultérieur : vérifier les annotations et les droits, harmoniser les libellés, regrouper aussi les acquisitions d'une même machine si l'identifiant était absent, fixer les splits par groupe avant fenêtrage et choisir une longueur commune. Les statistiques réseau doivent être apprises uniquement sur train. Les libellés libres ne deviennent pas automatiquement des classes du spécialiste MIMII.

## Tester le spécialiste existant

Ouvrir http://127.0.0.1:8765/test-model. La page charge le checkpoint réel local `artifacts/experiments/mimii_fan_real_v1/final.pt`. Quatre exemples MIMII (normal/anormal, validation/test) sont proposés, avec vérification du manifeste et du fichier. Ce sont des exemples déjà évalués, pas une nouvelle mesure de généralisation.

L'inférence sur fichier utilisateur exige exactement 16 000 échantillons, 16 kHz, huit canaux avec le contrat WAV du modèle. Aucun rééchantillonnage ni duplication mono/stéréo. La normalisation s'applique à une copie d'entrée ; les mesures physiques et le signal enregistré sont inchangés. Aucun poids ni seuil n'est modifié. Le score affiché est celui d'une fenêtre ; les résultats documentés par clip agrègent dix fenêtres.

Limite majeure : 100 % de faux positifs sur les clips normaux de la machine test lors de l'expérience initiale. Aucune confiance calibrée ni diagnostic mécanique. Voir REAL_TRAINING_V1.md.

## Validation

Tests : sauvegarde/relecture exacte des échantillons PCM, amplitude/RMS, idempotence, regroupement, refus des sélections invalides, endpoint explicite et origine locale, inférence sans mutation et contrat d'entrée strict. Les tests audio nécessitent FFmpeg/ffprobe ; les tests HTTP et le décodage utilisent le loopback local. Vérification navigateur sur une base de contrôle sous `artifacts/qa_learning/`, séparée de la base utilisateur.

Validation du 2026-10-28 : **115 tests réussis** sur une copie exacte de l’index préparé, sans les changements fusion non commités. L’ensemble du répertoire de travail avait également passé 120 tests. Dans le navigateur : import WAV de contrôle, sélection par glisser, enregistrement dans une base QA isolée et compteur relu ; affichage mobile vérifié. Le checkpoint réel conserve son SHA256 `e2ed2d746cabeed3b66bf5b77bf62e0fe22b04e1023182bd0c3efd59f7d94898`.
