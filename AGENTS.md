# Signal-LM — consignes communes

## Objectif
Construire un petit modèle de perception de signaux initialisé de zéro, puis
des spécialistes entraînés séparément. Aucun poids préentraîné externe.

## Règles de développement
- Lire docs/SPEC_V1.md avant de modifier les interfaces.
- Conserver la distinction mesures physiques / interprétation apprise.
- Ne jamais présenter des logits vierges comme un diagnostic ou une confiance.
- Les tenseurs publics sont [batch, samples, channels] ; les arrays [samples, channels].
- Ne jamais normaliser les mesures physiques avec les paramètres du réseau.
- Préserver les unités et la fréquence d'échantillonnage.
- Ne pas ajouter de données réelles, secrets ou poids dans Git.
- Ne pas remplacer de fichiers utilisateur ou de checkpoints sans instruction explicite.
- Tester les propriétés physiques (fréquence, amplitude, RMS), le rechargement des poids
  et le passage des gradients lorsqu'on modifie ces fonctions.
- Lancer `python -m pytest -q` et rapporter précisément les limites de validation.
- Codex et Claude Code travaillent séquentiellement ou sur des branches distinctes.
- Le lot initial ne comprend ni entraînement métier ni collecte de données réelles.

## Prochaines étapes
Pipeline d'apprentissage avec séparation par acquisition/machine, normalisation
apprise uniquement sur train, calibration et validation ; puis streaming et export.
