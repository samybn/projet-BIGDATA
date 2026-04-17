
# DétectInfo - Détecteur de Fake News (Mamba AI)

DétectInfo est une solution complète d'analyse de véracité d'informations basée sur l'architecture de pointe **Mamba (State Space Model)**. Le projet combine un modèle de Deep Learning haute performance avec une interface web moderne et sécurisée.

## 🚀 Fonctionnalités
- **Architecture Mamba** : Utilisation d'un modèle SSM (State Space Model) minimal pour une efficacité accrue sur les séquences textuelles.
- **Interface Premium** : Design moderne inspiré de ChatGPT/Claude (Dark Mode, Glassmorphism).
- **Gestion de Compte** : Système d'inscription et de connexion sécurisé avec base de données SQLite.
- **Analyse en Temps Réel** : Retour immédiat sur la fiabilité d'un titre ou d'un paragraphe avec score de confiance.

---

## 🛠️ Technologies Utilisées

### Intelligence Artificielle
- **Core Architecture** : Mamba Block (Custom Implementation)
- **Framework** : PyTorch
- **Traitement de texte (NLP)** : Scikit-learn (TF-IDF Vectorization), NLTK (Stopwords)
- **Optimisation** : Hyperparameter Search (RandomizedSearch concept)

### Web & Backend
- **Framework Web** : Flask
- **Base de Données** : SQLite avec Flask-SQLAlchemy
- **Authentification** : Flask-Login (Session management)
- **Frontend** : HTML5, Vanilla JavaScript, CSS3

---

## 📦 Installation (Étape par étape)

### 1. Prérequis
Assurez-vous d'avoir **Python 3.8+** installé sur votre machine.

### 2. Clonage / Préparation du dossier
Placez tous les fichiers du projet dans un dossier dédié (ex: `projet IA`).

### 3. Installation des dépendances
Ouvrez un terminal dans le dossier du projet et exécutez les commandes suivantes :

```powershell
# Installation des bibliothèques de manipulation de données et ML
python -m pip install torch pandas numpy scikit-learn tqdm joblib nltk

# Installation des bibliothèques pour le serveur Web
python -m pip install flask flask-cors flask-sqlalchemy flask-login
```

### 4. Téléchargement des ressources NLTK
Le projet utilise des listes de mots vides (stopwords) pour le nettoyage du texte. Elles seront téléchargées automatiquement au premier lancement, ou manuellement via :
```python
python -c "import nltk; nltk.download('stopwords')"
```

---

## 📖 Utilisation

### Étape 1 : Entraînement du modèle
Avant de pouvoir faire des prédictions, vous devez entraîner le modèle sur votre dataset (ex: `fake_news.csv`).
```powershell
python train_optimize.py
```
*Cette étape générera un fichier `models/mamba_fake_news.pth` contenant le modèle entraîné.*

### Étape 2 : Lancer l'interface Web
Une fois le modèle entraîné, lancez le serveur Flask :
```powershell
python app.py
```
Accédez ensuite à l'interface via votre navigateur à l'adresse :
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📂 Structure du Projet
- `mamba_minimal.py` : Définition de l'architecture Mamba et du Classifieur.
- `train_optimize.py` : Script d'entraînement et d'optimisation des hyperparamètres.
- `app.py` : Serveur Backend Flask & API.
- `predict.py` : Script utilitaire pour les prédictions en ligne de commande.
- `ui/` : Fichiers sources de l'interface frontend (HTML, CSS, JS).
- `models/` : Dossier de stockage du modèle (.pth) et du vectorizer.
- `users.db` : Base de données SQLite pour les comptes utilisateurs.

---

## 🛡️ Sécurité
Les mots de passe des comptes créés dans l'interface sont **hachés** (PBKDF2 avec sel) avant d'être enregistrés dans la base de données, garantissant qu'ils ne sont jamais stockés en texte clair.
