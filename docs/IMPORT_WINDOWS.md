# Import du premier lot sur Windows

Le dépôt GitHub était vide à la préparation. La publication depuis ChatGPT a
été refusée par GitHub (403 : Resource not accessible by integration).
Cette archive contient donc le projet vérifié, à importer depuis le poste local.

1. Extraire l'archive dans un dossier temporaire.
2. Copier le CONTENU de son dossier `Signal-LM` dans `E:\Signal-LM`.
   Ne pas écraser des fichiers existants sans les comparer. Ne pas créer
   involontairement `E:\Signal-LM\Signal-LM`.
3. Dans PowerShell, suivre les commandes adaptées ci-dessous.

```powershell
Set-Location E:\Signal-LM
git status
```

Si le dossier n'est pas encore un dépôt Git (et seulement dans ce cas) :

```powershell
git init -b main
git remote add origin https://github.com/YassineAbouch/Signal-LM.git
```

Si le dépôt existe déjà, vérifier `git remote -v` et garder sa configuration
si elle pointe vers le dépôt attendu. Inspecter les fichiers avant le commit :

```powershell
git add .gitignore README.md AGENTS.md CLAUDE.md pyproject.toml configs docs src tests
git diff --cached --stat
git commit -m "feat: add virgin signal model and deterministic analysis"
git push -u origin HEAD:main
```

Ne pas utiliser de force-push. Si le distant a reçu des commits entre-temps,
faire examiner l'état par Claude Code ou Codex avant de résoudre la divergence.
Le push utilise votre authentification GitHub locale ; ne partager aucun token.
Suivre ensuite le README pour installer Python/PyTorch et lancer les tests.
