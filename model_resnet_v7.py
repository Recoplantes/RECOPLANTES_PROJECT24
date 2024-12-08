#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour construire, entraîner et évaluer le modèle model_resnet_v7
présenté dans notre rapport de modélisation

"""
# Import des librairies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense

from keras.optimizers import Adam
from keras import regularizers
from keras.metrics import AUC, Precision, Recall, F1Score

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from sklearn.metrics import confusion_matrix

# Chemins d'accès aux datasets train, val et test

data_train_path = "Data/color_split/train"
data_val_path = "Data/color_split/val"
data_test_path = "Data/color_split/test"

# Générateurs d'images

train_data_generator = ImageDataGenerator(
    # Preprocessing = images from RGB to BGR, zero-center each color channel, without scaling.
    preprocessing_function = preprocess_input,
    #rescale=1./ 255,
    # data augmentation
    rotation_range=30,  # Rotation aléatoire de l'image
    width_shift_range=0.2,  # Décalage horizontal
    height_shift_range=0.2,  # Décalage vertical
    zoom_range=0.3,  # Zoom avant ou arrière
    horizontal_flip=True,  # Renversement horizontal
    vertical_flip=True,  # Renversement vertical
    brightness_range=[0.9, 1.1,],  # Variation de la luminosité
    channel_shift_range=20  # Variations de couleur
)
val_data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    #rescale=1./ 255
)
test_data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    #rescale=1./ 255
)

# Itérateurs pour générer les données par lot

batch_size = 32
img_height = 224
img_width = 224

train_generator = train_data_generator.flow_from_directory(
      directory=data_train_path,    # répertoire des images
      class_mode ="categorical",    # labels = vecteurs one-hot encoded
      target_size = (img_height,img_width),   # redimensionnement des images
      batch_size = batch_size
)
val_generator = val_data_generator.flow_from_directory(
      directory=data_val_path,
      class_mode ="categorical",
      target_size = (img_height,img_width),
      batch_size = batch_size,
      shuffle=False
)
test_generator = test_data_generator.flow_from_directory(
      directory=data_test_path,
      class_mode ="categorical",
      target_size = (img_height,img_width),
      batch_size = batch_size,
      shuffle=False
)

# Callbacks

# Instanciation de la classe CustomCheckpoint
# = callback personnalisé pour enregistrer le modèle toutes les n epochs

# Sauvegarde de la date du jour dans current_date
from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d")
print("Date pour le nommage des fichiers de sauvegarde :", current_date)

class CustomCheckpoint(Callback):
    def __init__(self, save_freq):
        self.save_freq = save_freq  # Fréquence de sauvegarde (en epochs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:  # Sauvegarde toutes les `save_freq` epochs
            filepath = f"model_resnet_{current_date}_epoch_{epoch + 1:02d}.keras"
            self.model.save(filepath)
            print(f"\n Modèle sauvegardé dans {filepath}")


# Instanciation de la classe SaveTrainingHistoryCallback
# = callback personnalisé pour enregistrer l'historique d'entraînement au format csv


class SaveTrainingHistoryCallback(Callback):
    def __init__(self, save_freq=1, save_path=f"model_resnet_historique_entrainement_{current_date}.csv"):
        super(SaveTrainingHistoryCallback, self).__init__()
        self.save_freq = save_freq  # Fréquence de sauvegarde (en epochs)
        self.save_path = save_path  # Chemin pour sauvegarder l'historique
        self.history = {}  # Dictionnaire pour accumuler les métriques

    def on_epoch_end(self, epoch, logs=None):
        # Ajouter les métriques actuelles à l'historique
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)  # Accumulate values

        # Sauvegarder l'historique tous les `save_freq` epochs
        if (epoch + 1) % self.save_freq == 0:

            min_length = min(len(values) for values in self.history.values())
            filtered_history = {key: values[:min_length] for key, values in self.history.items()}

            history_df = pd.DataFrame(filtered_history)
            history_df.to_csv(self.save_path, index=False)
            print(f"\n Historique sauvegardé après {epoch + 1} époques dans {self.save_path}")

# Instanciation des callbacks

checkpoint_best = ModelCheckpoint(
    filepath=f"model_resnet_best_{current_date}.keras",  # Nom du fichier
    monitor="val_accuracy",  # métrique surveillée
    save_best_only=True,  # enregistre uniquement le meilleur modèle
    mode="max",  # sauvegarder le modèle quand l'accuracy est maximale
    save_weights_only=False,  # enregistre tout le modèle
    verbose=1,  # Affiche un message quand le modèle est enregistré
)

checkpoint_freq = CustomCheckpoint(
    save_freq=10,  # fréq d'enregistrement du modèle en epochs
)

history_callback = SaveTrainingHistoryCallback(
    save_freq=5,  # fréq d'enregistrement de l'historique en epochs
    save_path=f"model_resnet_historique_entrainement_{current_date}.csv",
)


"""
Transfer Learning v7

Entraînememnt lancé sur 100 epochs après dégel de l'entièreté de ResNet50
Taille des images fournies au modèle pour l'entraînement modifiée de 64x64 à 224x224 pixels
"""

# Modèle ResNet50 pour la classification d'image

base_model = ResNet50(
    weights='imagenet',  # Poids pré-entraînés sur ImageNet
    include_top=False,   # Exclusion des couches fully connected
    input_shape=(img_height, img_width, 3)
)

# Dégel des dernières couches de ResNet50

for layer in base_model.layers:  # Dégel de l'entièreté du modèle
    layer.trainable = True

# Instanciation du modèle : model_resnet

model_resnet = Sequential()

model_resnet.add(base_model)
model_resnet.add(GlobalAveragePooling2D())

model_resnet.add(Dense(units=256, activation='relu'))
model_resnet.add(Dense(units=128, activation='relu'))
model_resnet.add(Dense(units=64, activation='relu'))
model_resnet.add(Dense(units=38, activation="softmax"))

model_resnet.summary()

# Compilation du modèle : model_resnet

model_resnet.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'Adam',
    metrics = ['accuracy']
)

# Entraînement du modèle : model_resnet

epochs = 100

model_resnet_history = model_resnet.fit(train_generator,
                                        epochs = epochs,
                                        validation_data = val_generator,
                                        callbacks = [checkpoint_freq,
                                                     checkpoint_best,
                                                     history_callback],
                                        verbose=True
                                        )

# Evaluation graphique de l'entraînement

# précisions d'entraînement et de validation obtenues pendant l'entraînement
train_acc_resnet = model_resnet_history.history['accuracy']
val_acc_resnet = model_resnet_history.history['val_accuracy']

# Courbe de la précision sur l'échantillon d'entrainement
plt.plot(np.arange(1 , epochs+1, 1),
         train_acc_resnet,
         label = 'ResNet Training Accuracy',
         color = 'red')

# Courbe de la précision sur l'échantillon de validation
plt.plot(np.arange(1 , epochs+1, 1),
         val_acc_resnet,
         label = 'ResNet Validation Accuracy',
         color = 'lightcoral')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluation sur l'ensemble de test depuis un modèle chargé

# Chargement du modèle à évaluer dans classifier_resnet

model_path = 'model_resnet_v7_best.keras' # chemin du modèle à charger

classifier_resnet = keras.load_model(model_path)

# Scores

print("\n\nScores de model_resnet sur l'ensemble de test :", "\n\n")

score_resnet = classifier_resnet.evaluate(test_generator)

print("\n\nScores : \n\n", score_resnet, end="\n\n")


# Predictions sur l'ensemble de test

# Prédictions sous forme de probas pour chaque classe
predictions_resnet_probas = classifier_resnet.predict(test_generator)

# Transformation des probabilités en classes prédites
predictions_resnet_classes = np.argmax(predictions_resnet_probas, axis = 1)

# Classes issues de l'ensemble test
true_classes = test_generator.classes

# Labels des classes (= noms des dossiers)
class_labels = list(test_generator.class_indices.keys())


# Affichage de la matrice de confusion

cm = confusion_matrix(true_classes, predictions_resnet_classes)

print("\n\nPrédictions sur l'ensemble de test et matrice de confusion")
plt.figure(figsize=(15, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Matrice de Confusion")
plt.ylabel("Classes Réelles")
plt.xlabel("Classes Prédites")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Affichage de la matrice de confusion normalisée

print("\n\nPrédictions sur l'ensemble de test et matrice de confusion normalisée")

plt.figure(figsize=(20, 20))

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels)

plt.title("Matrice de Confusion normalisée")
plt.ylabel("Classes Réelles")
plt.xlabel("Classes Prédites")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()