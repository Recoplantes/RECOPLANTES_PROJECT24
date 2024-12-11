#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ce script contient le code pour exécuter le modèle TEM3, un modèle CNN avec plusieurs couches Conv2D et des 
techniques de régularisation comme le Dropout et la régularisation L2. Le modèle a été entraîné sur l'ensemble des 
données d'entraînement et de validation, et inclut les poids du modèle, la data augmentation, ainsi que tous les 
paramètres nécessaires à son exécution. Ce script permet de créer et d'entraîner un modèle de classification 
d'images avec TensorFlow et Keras.

Étapes principales :
1. Importation des packages.
2. Initialisation de la graine aléatoire pour la reproductibilité des résultats.
3. Configuration de la Data Augmentation.
4. Création des générateurs de données pour l'entraînement, la validation et les tests.
5. Création du modèle CNN avec régularisation L2 et Dropout.
6. Chargement des poids préalablement enregistrés et résumé du modèle pour vérification.

Le modèle utilise l'optimiseur 'adam' et est conçu pour une classification multi-classes avec 38 classes de sortie.
"""

# Importation des packages
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Fixer la graine aléatoire pour assurer la reproductibilité
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

#Les chemins d'accès

data_train_path = 'Data/color_split/train'
data_val_path = 'Data/color_split/val'
data_test_path = 'Data/color_split/test'

# Configuration de la Data Augmentation
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_data_generator = ImageDataGenerator(rescale=1./255)

test_data_generator = ImageDataGenerator(rescale=1./255)

# Création des générateurs de données
batch_size = 32
img_height = 64
img_width = 64

train_generator = train_data_generator.flow_from_directory(
    directory='data_train_path',  
    class_mode='categorical',
    target_size=(img_height, img_width),
    batch_size=batch_size
)

val_generator = val_data_generator.flow_from_directory(
    directory='data_val_path',
    class_mode='categorical',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

test_generator = test_data_generator.flow_from_directory(
    directory='data_test_path',
    class_mode='categorical',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Création du modèle TEM3 avec régularisation L2

    model_TEM3 = Sequential()

    # Couche d'entrée
    model_TEM3.add(Input(shape=(img_height, img_width, 3), name='Input'))

    # Première couche Conv2D + régularisation L2
    model_TEM3.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid',
                          kernel_regularizer=regularizers.l2(0.01)))
    model_TEM3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Deuxième couche Conv2D + régularisation L2
    model_TEM3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid',
                          kernel_regularizer=regularizers.l2(0.01)))
    model_TEM3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Troisième couche Conv2D + régularisation L2
    model_TEM3.add(Conv2D(filters=94, kernel_size=(3, 3), activation='relu', padding='valid',
                          kernel_regularizer=regularizers.l2(0.01)))
    model_TEM3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Dropout pour éviter l'overfitting
    model_TEM3.add(Dropout(rate=0.2))

    # Couche Flatten
    model_TEM3.add(Flatten())

    # Couche Dense 128 unités + régularisation L2
    model_TEM3.add(Dense(units=128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01)))

    # Couche de sortie Dense 38 unités
    model_TEM3.add(Dense(units=38, activation='softmax'))

    model_TEM3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Compiler le modèle 
model_TEM3.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Entraînement avec sauvegarde

model_TEM3.fit(train_generator,
                epochs=50,
                validation_data=val_generator,
                callbacks=[checkpoint])

# Charger les poids depuis la racine du projet
weights_path = "leila_best_model_cnn_TEM3.keras"  
if os.path.exists(weights_path):
    model_TEM3.load_weights(weights_path)

# Résumer le modèle pour vérifier qu'il est prêt
model_TEM3.summary()

#Test du model

# Labels des classes (= noms des dossiers)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Image à tester
path = 'Data/images_pred/Apple__Apple_scab/apple_scab.jpg'

#Chargement du model 

model_path = ""leila_best_model_cnn_TEM3.keras" 

# chemin du modèle à charger

classifier_cnn_TEM3 = keras.load_model(model_path)

# Charger et prétraiter l'image
test_image = load_img(path, target_size=(64, 64))  # Redimensionner l'image pour qu'elle corresponde à la taille d'entrée attendue par le modèle
test_image = test_image.convert('RGB')
test_image = img_to_array(test_image)  # Convertir l'image en tableau numpy
test_image = test_image / 255.0  # Normaliser l'image en divisant par 255.0

# Prédiction de probabilité pour chaque classe
proba = model_TEM3.predict(np.expand_dims(test_image, axis=0))[0]  # Prédiction sur l'image d'entrée
predicted_class_idx = np.argmax(proba)  # Indice de la classe prédite
predicted_proba = round(100 * proba[predicted_class_idx], 2)  # Probabilité de la classe prédite

# Extraire le nom de la culture et de la maladie à partir du nom de la classe prédite
predicted_class_name = class_names[predicted_class_idx]
culture_name, disease_name = predicted_class_name.split('___')[0], predicted_class_name.split('___')[1]

# Affichage des résultats

img = mpimg.imread(path)
plt.axis('off')
plt.text(-10, -15, f'Culture: {culture_name}\nMaladie: {disease_name}\nIndice de confiance: {predicted_proba}%', 
         color="#00897b", fontsize=14, fontweight='bold')
plt.imshow(img)
plt.show()
