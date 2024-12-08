#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour réaliser des predictions à partir d'un modèle entraîné.

Preprocessing calibré pour réaliser des prédictions avec model_resnet_v7_best :
- fonction de peprocessing propre au modèle ResNet-50
- taille attendue des images en entrée : 224x224 pixels

"""

# Chargement dans 'classifier' du modèle pour la prédiction

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model_path = 'model_resnet_v7_best.keras' # chemin du modèle à charger

classifier = load_model(model_path)

# Import des librairies

import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Chemins d'accès aux datasets train, val et test

data_train_path = "Data/color_split/train"
data_val_path = "Data/color_split/val"
data_test_path = "Data/color_split/test"

# Labels des classes (= noms des dossiers)

class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Fonction pour la prédiction et l'affichage des résultats

def predict_plant_disease(classifier, image_path):
    """
    Prédit la maladie d'une plante à partir d'une image.

    Args:
        classifier: Modèle chargé pour effectuer la prédiction.
        image_path (str): Chemin vers l'image à prédire.

    Returns:
        Aucun. Affiche les résultats sous forme graphique et texte.
    """
    # Prétraitement de l'image
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = test_image.convert('RGB')  # Convertit en RGB pour s'assurer d'avoir 3 canaux
    test_image = image.img_to_array(test_image)
    test_image = preprocess_input(test_image)  # Prétraitement spécifique au modèle
    test_image = np.expand_dims(test_image, axis=0)  # Ajoute une dimension pour simuler un batch

    # Prédiction
    proba = classifier.predict(test_image)[0]
    predicted_class_idx = np.argmax(proba)
    predicted_proba = round(100 * proba[predicted_class_idx], 2)

    # Nom de la culture et de la maladie
    predicted_class_name = class_names[predicted_class_idx]
    culture_name, disease_name = predicted_class_name.split('___')

    # Extraction du nom du fichier et du dossier parent (classe réelle)
    file_name = os.path.basename(image_path)
    real_class_name = os.path.basename(os.path.dirname(image_path))

    # Affichage des résultats
    img = mpimg.imread(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.text(0, -90, 'Prédiction',
             fontsize=9, color='black', fontweight='bold')
    plt.text(0, -40, f'Culture: {culture_name}\nMaladie: {disease_name}\nIndice de confiance: {predicted_proba}%',
             color="#00897b", fontsize=14, fontweight='bold')
    plt.text(0, -10, f"Nom du fichier : {file_name}\nClasse réelle : {real_class_name}",
             fontsize=8, color='gray', fontweight='bold')
    plt.show()


# Chemin de l'image à prédire et appel de la fonction de prédiction

image_path = "Data/color_split/test/Potato___Early_blight/12a7bc09-8d7f-4790-8046-6e3f12f71399___RS_Early.B 7973.JPG"

predict_plant_disease(classifier, image_path)

# Prédire plusieurs images issues du web

image_paths = [
    "Data/images_pred/Apple___Apple_scab/Apple+Scab+(2).jpg",
    "Data/images_pred/Apple___Apple_scab/Apple-Scab-J751-18-2.jpg",
    "Data/images_pred/Apple___Apple_scab/apple-scab-apple-1710342719.jpg",
    "Data/images_pred/Apple___Apple_scab/apple_scab.jpg",
    "Data/images_pred/Apple___Apple_scab/apple_scab_02.jpg",
    "Data/images_pred/Apple___Apple_scab/guy_deguire_tavelure_fait-03.jpg",
    "Data/images_pred/Apple___Apple_scab/img_1916.jpg",
    "Data/images_pred/Apple___Apple_scab/pommier___tavelure_sqr.jpg"
]

for image_path in image_paths:
    predict_plant_disease(classifier, image_path)

