# Projet : Évaluation de la Satisfaction des Passagers des Compagnies Aériennes

## Description

Ce projet a pour objectif d’évaluer la satisfaction d’un passager d’une compagnie aérienne, en se basant sur plusieurs critères. Les données sont initialement stockées dans une base relationnelle puis transformées et nettoyées pour alimenter une base analytique. Des modèles de machine learning sont entraînés afin de prédire la satisfaction (0 ou 1), et une application web est mise en place pour permettre aux utilisateurs d’entrer leurs informations et d’obtenir une prédiction de satisfaction.

## Fonctionnalités

- **Préparation des données** :  
  - Renommage des colonnes  
  - Nettoyage des données (suppression ou interpolation des valeurs manquantes)  
  - Détection et correction des aberrations  
  - Combinaison des retards de départ et d’arrivée en une seule variable  
  - Étiquetage binaire de la satisfaction (0/1)  
  - Suppression des colonnes non représentatives  
  - Encodage des variables catégorielles (one-hot encoding)
  
- **Modélisation** :  
  - Implémentation et comparaison de différents modèles de machine learning  
  - Optimisation des hyperparamètres  
  - Analyse des métriques de performance (précision, rappel, f1-score, etc.)

- **Application Web** :  
  - Intégration du modèle de machine learning dans une application web  
  - Formulaire interactif pour l’utilisateur  
  - (Optionnel) Système d’authentification (Login/Register) et gestion de session  
  - Respect des règles de sécurité et des bonnes pratiques d’architecture

- **Organisation** :  
  - Mode Agile (Kanban)  
  - Documentation des choix et des méthodes  
  - Répartition claire des tâches entre les membres de l’équipe

## Prérequis

- Python 3.9 
- Bibliothèques Python :  
  - `pandas`  
  - `numpy`  
  - `scikit-learn`  
  - `matplotlib` / `seaborn` (pour la visualisation des données)  
  - `flask` ou `streamlit` (pour l’application web)  
- Outil de gestion de base de données : MySQL, PostgreSQL, ou autre (selon le choix du groupe)  
- Git/GitHub pour la gestion de versions

## Installation

1. **Cloner le dépôt Git** :  
   ```bash
   git clone https://github.com/votre-equipe/projet-satisfaction-passagers.git
   cd projet-satisfaction-passagers
   ```

2. **Installer les dépendances** :  
   Nous vous recommandons l’utilisation d’un environnement virtuel.  Version PYTHON 3.9
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Sous Linux/Mac
   venv\Scripts\activate      # Sous Windows
   pip install -r requirements.txt
   ```

3. **Configuration de la base de données** :  
   - Créez et initialisez les bases de données (relationnelle et analytique) selon la documentation interne.  
   - Mettez à jour les identifiants de connexion dans le fichier de configuration (`config.py` ou `.env`).

## Exécution

1. **Préparation des données** :  
   - Lancez le script d’ingestion et de nettoyage des données :  
     ```bash
     python data_preprocessing.py
     ```
   
2. **Entraînement des modèles** :  
   - Exécutez le script d’entraînement et de test des modèles :  
     ```bash
     python train_model.py
     ```
   
3. **Lancement de l’application web** :  
   - Démarrez l’application :  
     ```bash
     python app.py
     ```
   - Rendez-vous sur `http://localhost:5000` (par défaut) pour interagir avec l’interface.

## Structure du Dépôt

```
projet-satisfaction-passagers/
│
├─ data/                     # Données sources (non versionnées si volumineuses)
│
├─ notebooks/                # Notebooks Jupyter pour l’exploration des données 
│
├─ scripts/                  # Scripts annexes (outils, tests, etc.)
│
├─ src/
│   ├─ data_preprocessing.py # Script de nettoyage et préparation des données
│   ├─ model.py              # Définition du modèle de ML
│   ├─ train_model.py        # Script d’entraînement et d’évaluation
│   ├─ utils/                # Fonctions utilitaires
│
├─ app.py                    # Application web
├─ requirements.txt          # Liste des dépendances
├─ README.md                 # Ce document
└─ config.py                 # Fichier de configuration (ou .env)
```

## Documentation et Justifications

- **Document d’architecture et de choix technologiques** : disponible dans le répertoire `docs/`, expliquant les décisions de modélisation, de stockage et de déploiement.
- **Kanban & Répartition des tâches** : un lien vers l’outil de gestion de projet (ex: Trello, Jira, GitHub Projects) sera communiqué dans la documentation interne.
- **Agilité** : les sprints, les user stories, les rétrospectives et les objectifs sont détaillés dans `docs/`.

## Améliorations Futures

- Intégration d’autres modèles de machine learning plus sophistiqués (XGBoost, LightGBM, etc.)
- Implémentation d’un CI/CD pour le déploiement automatisé.
- Amélioration du front-end de l’application.
- Intégration d’un système d’authentification complet (Login/Register) avec stockage des sessions.
- Mise en place d’une API REST pour la prédiction.

---

## Auteurs
- José el patron
- Jules el puma
- Yann el aguila

## Actualización de pip

Pour actualizar `pip`, ejecuta el siguiente comando:
```bash
python.exe -m pip install --upgrade pip
```