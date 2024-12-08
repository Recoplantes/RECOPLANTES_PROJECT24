# Projet de Reconnaissance de Plantes avec TensorFlow et Keras

## Description du projet

Ce projet utilise **TensorFlow** et **Keras** pour construire un modèle de reconnaissance d'images de plantes basé sur un réseau de neurones convolutifs (CNN). L'objectif principal est de former un modèle capable de reconnaître et de classer des images en fonction de certaines caractéristiques. Ce projet implique la manipulation d'images, le prétraitement des données et la construction d'un modèle de classification.

### Objectifs

- Prétraiter les images avec **ImageDataGenerator** de Keras.
- Créer un modèle de classification basé sur un **réseau de neurones convolutifs (CNN)**.
- Utiliser **ModelCheckpoint** pour sauvegarder le meilleur modèle pendant l'entraînement.
- Gérer les défis liés à l'installation et la compatibilité des bibliothèques, en particulier avec **TensorFlow**.

## Prérequis

Avant de commencer, vous devez vous assurer que vous avez installé Python 3.x et les bibliothèques suivantes :

- **TensorFlow 2.x** (assurez-vous de vérifier la compatibilité de la version avec votre système)
- **NumPy**
- **Matplotlib**
- **Keras**
- **scikit-learn**
- **PIL (Pillow)**

Pour le modèle TEM3
### Installation

1. Clonez ce projet dans votre répertoire local :
```bash
   git clone https://github.com/Recoplantes/projet_recom_plantes_TEM3_.git

2. Installez les dépendances dans un environnement virtuel (si vous n'en avez pas déjà un) :
    python -m venv env

3. Activez votre environnement virtuel :
    Windows : .\env\Scripts\activate
    Mac/Linux: source env/bin/activate

4. Installez les bibliothèques nécessaires avec pip :
    pip install -r requirements.txt

5. Préparez vos données d'images dans un répertoire dédié.

6. Exécutez le script principal pour entraîner le modèle ( Ce script contient la totalité des éléments necessaire a la bonne execution du code):

    python main_TEM3.py

Remarque: La data se trouve dans un dossier différent, nous avons utiliser des chemins d'accès pour permettre la bonne execution du code
