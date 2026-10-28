# Signal-LM — spécification V1

## But et périmètre
Une famille de modèles compacts de perception des signaux, initialisés de zéro.
Un spécialiste sera une configuration, un prétraitement, des poids entraînés et
une évaluation propres à un domaine. Ce n'est pas un modèle de langage.
Le lot 1 fournit le socle vierge, les mesures physiques et les tests synthétiques.

## Contrat d'entrée
- Array [échantillons, canaux] ; un vecteur 1D devient un seul canal.
- Fréquence en Hz strictement positive, échantillons finis, minimum 4 points.
- Noms uniques et unités par canal. Une unité absente est `unspecified`.
- CSV : en-tête obligatoire, colonnes numériques explicitement sélectionnées,
  séparateur configurable, point décimal. La fréquence est fournie explicitement.
- Le caller garantit un échantillonnage régulier et des canaux synchronisés.
  Les timestamps CSV ne sont ni interprétés ni validés dans ce lot.
- Pas d'imputation, rééchantillonnage ou mélange de capteurs à fréquences différentes implicites.
- Quatre points constituent une limite logicielle, pas une durée suffisante pour
  un diagnostic ; la fenêtre dépendra du phénomène et du spécialiste.

## Mesures déterministes
Moyenne, RMS total, RMS sans composante continue, pic absolu, crête-à-crête.
FFT réelle après soustraction de la moyenne et fenêtre Hann périodique.
Spectre d'amplitude unilatéral corrigé du gain cohérent ; DC et Nyquist
ne sont pas doublés. La fréquence dominante exclut DC et vaut null si le
signal centré est nul. Le pas fréquentiel vaut fs/N ; ce n'est pas à lui seul
le pouvoir de résolution réel d'une fenêtre. Une sinusoïde hors-bin entraîne
fuite spectrale et biais d'amplitude. Pas de PSD ou de spectrogramme dans ce lot.
Le spectre est optionnel car il peut produire un JSON volumineux.
Le résultat expose `prediction.status=unavailable`, jamais une classe inventée.

## Réseau vierge
CNN 1D : convolutions de largeur 7, canaux 16/32/64, GELU,
pooling moyen temporel et projection en 64 dimensions. Aucun vocabulaire.
Pas de sous-échantillonnage pour ce premier socle. Champ réceptif local limité
à 19 échantillons avant pooling : cette base n'est pas validée pour des motifs
longs ; dilatations, plusieurs échelles ou spectrogrammes seront évalués ensuite.
Les lots doivent avoir une longueur commune ; pas de padding masqué implémenté.
Tête de classification facultative et dimensionnée avant spécialisation.
Par défaut aucune tête, donc aucun nombre de classes arbitraire.
Les dimensions et la graine sont configurables. Les poids sont initialisés
aléatoirement avec PyTorch, sans import de poids tiers.
Les biais peuvent avoir des valeurs nulles : « vierge » signifie non entraîné,
pas que tous les paramètres doivent être non nuls.

## Checkpoint
Version du format, état `untrained`, nombre d'étapes 0, configuration et state_dict.
L'initialisation ne remplace jamais un fichier existant. Le rechargement se fait
sur CPU via `weights_only=True`, puis le modèle peut être déplacé sur le GPU.
La métadonnée untrained décrit notre procédure, elle n'est pas une preuve
cryptographique de provenance d'un checkpoint externe.

## Spécialisation ultérieure
Ne pas confondre poids vierges et poids spécialisés. Le format spécialisé devra
contenir fréquence, unités, fenêtrage, normalisation, classes, provenance des données,
seuils éventuels, statut entraîné et résultats d'évaluation.
Classification multi-classe, multi-label et anomalie ne sont pas interchangeables.
Chaque objectif aura sa perte et sa validation. La confiance et le rejet inconnu
exigeront une calibration ; une softmax n'est pas une garantie de fiabilité.
Découper train/validation/test par acquisition, période ou machine AVANT de
fabriquer des fenêtres voisines. Évaluer les signaux faibles par niveau de bruit.

## Jalons
1. Présent : arrays/CSV, mesures, modèle vierge, checkpoint, tests.
2. Apprentissage et validation d'une spécialité, avec données choisies plus tard.
3. Streaming à fenêtres et tampon, contrôle des ruptures et de la latence.
4. Évaluation d'autres architectures, quantification/export et matériel cible.

## Critères d'acceptation du lot 1
La démo 120 Hz donne 120 Hz et RMS = 1/sqrt(2).
Les amplitudes paires/impaires et Nyquist sont testées.
Le multi-canal et les entrées invalides sont vérifiés.
Sauvegarde/rechargement conserve exactement les sorties.
Les gradients passent ; aucun apprentissage d'expertise n'est revendiqué.
Le diagnostic système expose la disponibilité CUDA sans la présumer.

## Jalon 2 — pipeline spécialisé sur manifestes (branche data-prep)

Le modèle vierge et son checkpoint restent inchangés. Le package `training`
consomme les manifestes du lot données, sans modifier leur schéma, générateurs
ou amplitudes. Un entraînement concerne un seul couple `(specialist, task_type)`
et un seul contrat fréquence / canaux / unités / longueur. Le choix explicite
`parents` ou `windows` empêche de compter simultanément parents et fenêtres.
Le fenêtrage du lot données doit précéder le chargement si les longueurs diffèrent.

Les variantes spécialisées sont définies séparément :

- Classification exclusive : tête linéaire du CNN existant, CrossEntropy.
- Multi-label : même tête, BCEWithLogits, seuil 0,5 fixé avant évaluation.
- Événements : backbone des trois convolutions existantes, tête Conv1d de
  noyau 1 vers les labels temporels ; pas de pooling/projection pour cette
  variante. Sortie `[batch, samples, labels]`, masque binaire `[start, end)`,
  BCEWithLogits ; l'absence d'événement correspond à un masque nul. Champ
  réceptif inchangé de 19 échantillons. Les métriques sont au niveau échantillon,
  pas une validation des événements, débuts/fins ou délais de détection.
- Anomalie : encodeur existant sans tête de classes ; centre d'embedding fixé
  à partir des références train, puis réduction de la distance au centre sur
  références train seules (baseline une-classe). Seuil = quantile préfixé des
  scores des références validation. Biais/collapse possibles ; cette baseline
  n'est pas validée pour l'industrie et ne remplace pas une comparaison de méthodes.

La normalisation par canal est ajustée uniquement sur train (références seules
pour l'anomalie), puis conservée dans le checkpoint. Aucun tableau source n'est
modifié ; les mesures physiques continuent d'utiliser les données originales.
Les espaces de labels sont dérivés du train seulement. Une étiquette inconnue
hors train entraîne une erreur explicite ; aucun rejet inconnu calibré n'est fourni.

Checkpoint spécialisé : `format=signal_lm.specialist.v1`, `status=trained`,
configuration du modèle et de la tâche, poids, labels, contrat d'entrée,
normalisation, centre/seuil éventuels, étapes, sélection sur validation,
identifiant et graine du jeu, source(s), licence(s), empreintes SHA-256 du
manifeste/config/split_summary et résultats train/validation. Relecture CPU
avec `weights_only=True` ; refus d'écraser un fichier existant.

`fit` ne charge aucun signal test et ne produit aucun résultat test. La commande
séparée `evaluate` charge le checkpoint figé et vérifie la provenance avant de
joindre les résultats test à une nouvelle copie du checkpoint. Elle ne modifie
ni les poids, ni les statistiques, ni les seuils. Les résultats sont ventilés
par split, SNR cible (propre/−10/−5/0/5/10/20 dB selon le jeu), règle de holdout
et machine. Une softmax ou une distance au centre ne constitue pas une confiance.
La validation sert à la sélection et, pour l'anomalie, au seuil ; aucune
calibration indépendante de confiance n'est revendiquée.

Les essais automatiques utilisent le synthétique. L'import MIMII n'entraîne rien ;
un entraînement sur données réelles requiert une option explicite supplémentaire.
