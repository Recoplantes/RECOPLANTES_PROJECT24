# Projet de Reconnaissance de Plantes avec TensorFlow et Keras

## Description du projet

Ce projet utilise **TensorFlow** et **Keras** pour construire un modèle de reconnaissance d'images de plantes basé sur un réseau de neurones convolutifs (CNN). L'objectif principal est de former un modèle capable de reconnaître et de classer des images en fonction de certaines caractéristiques. Ce projet implique la manipulation d'images, le prétraitement des données et la construction d'un modèle de classification.

**Données source :**  
kaggle  
**PlantVillage Dataset**  
Dataset of diseased plant leaf images and corresponding labels  
https://www.kaggle.com/abdallahalidev/plantvillage-dataset  

## 1. Modèle TEM3
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
   git clone https://github.com/Recoplantes/RECOPLANTES_PROJECT24.git

2. Creez un enironnement virtuel :
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



## 2. Modèle model_cnn_4  

- Script **model_cnn_4.py** : Script pour construire, entraîner et évaluer le modèle model_cnn_4 présenté dans notre rapport de modélisation  

- Modèle **model_cnn_4_best.keras** : Modèle préentraîné qui peut être utilisé pour faire des prédictions.

Pas de preprocessing spécifique  

Taille attendue des images en entrée : 64x64 pixels  

   
## 3. Modèle model_resnet_v7
   
- Script **model_resnet_v7.py** : Script pour construire, entraîner et évaluer le modèle model_resnet_v7 présenté dans notre rapport de modélisation.  
   
- Script **predictions_resnet_v7.py** : Script pour réaliser des predictions à partir d'un modèle entraîné.  

Preprocessing calibré pour réaliser des prédictions avec **model_resnet_v7_best.keras** :  
fonction de peprocessing propre au modèle ResNet-50  
taille attendue des images en entrée : 224x224 pixels  

Note 1 : Le modèle préentraîné **model_resnet_v7.keras** est disponible en téléchargement (280 Mo) au lien qui suit.  
https://drive.google.com/file/d/1ghV5R5nnPlkq4gT8CVVLv3cy6jXGbGT8/view?usp=sharing  

Note 2 : des images pour essais de prédiction, issues du web, se trouvent dans le répertoire **Data/images_pred/Apple___Apple_scab** à la racine de ce dépôt

## 4. Script split_folders.py

Script pour diviser notre dataset en ensembles d'entraînement, validation et test.  
Basé sur la librairie splitfolders.  

Le répertoire "Data/plantvillage_dataset/color" contenant le dataset doit se trouver au même niveau que ce script.

Deux opérations sont réalisées :
- Split du dataset selon le ratio 0.8, 0.1, 0.1 dans le répertoire de sortie "color_split"  
- Split du dataset avec un nombre d'images fixes par ensemble 100, 25, 25 dans le réperoire de sortie "color_split_light"


