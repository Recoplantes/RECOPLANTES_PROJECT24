#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour construire, entraîner et évaluer le modèle model_cnn_4
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
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import load_model
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint

from sklearn.metrics import confusion_matrix

# Chemins d'accès aux datasets train, val et test

data_train_path = "Data/color_split/train"
data_val_path = "Data/color_split/val"
data_test_path = "Data/color_split/test"

# Générateurs d'images pour les ensembles train, val et test

train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
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
val_data_generator = ImageDataGenerator(rescale=1.0 / 255)
test_data_generator = ImageDataGenerator(rescale=1.0 / 255)

# Itérateurs pour générer les données par lot

batch_size = 32
img_height = 64
img_width = 64

train_generator = train_data_generator.flow_from_directory(
    directory=data_train_path,  # répertoire des images
    class_mode="categorical",  # labels = vecteurs one-hot encoded
    target_size=(img_height, img_width),  # redimensionnement des images
    batch_size=batch_size,
)
val_generator = val_data_generator.flow_from_directory(
    directory=data_val_path,
    class_mode="categorical",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
)
test_generator = test_data_generator.flow_from_directory(
    directory=data_test_path,
    class_mode="categorical",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
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
            filepath = f"model_cnn4_{current_date}_epoch_{epoch + 1:02d}.keras"
            self.model.save(filepath)
            print(f"\n Modèle sauvegardé dans {filepath}")


# Instanciation de la classe SaveTrainingHistoryCallback
# = callback personnalisé pour enregistrer l'historique d'entraînement au format csv


class SaveTrainingHistoryCallback(Callback):
    def __init__(self, save_freq=1, save_path=f"model_cnn_4_historique_entrainement_{current_date}.csv"):
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
    filepath=f"model_cnn_4_best_{current_date}.keras",  # Nom du fichier
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
    save_path=f"model_cnn_4_historique_entrainement_{current_date}.csv",
)

# Instanciation du modèle cnn : model_cnn_4

model_cnn_4 = Sequential()

model_cnn_4.add(Input(shape=(img_height, img_width, 3)))

model_cnn_4.add(Conv2D(filters=16, kernel_size=(3, 3), padding="SAME", activation="relu"))
model_cnn_4.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_4.add(Dropout(rate=0.2))

model_cnn_4.add(Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation="relu"))
model_cnn_4.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_4.add(Dropout(rate=0.2))

model_cnn_4.add(Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation="relu"))
model_cnn_4.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_4.add(Dropout(rate=0.2))

model_cnn_4.add(Conv2D(filters=128, kernel_size=(3, 3), padding="SAME", activation="relu"))
model_cnn_4.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_4.add(Dropout(rate=0.2))

model_cnn_4.add(Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation="relu"))
model_cnn_4.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_4.add(Dropout(rate=0.2))

model_cnn_4.add(Flatten())

model_cnn_4.add(Dense(units=256, activation="relu"))
model_cnn_4.add(Dense(units=128, activation="relu"))
model_cnn_4.add(Dense(units=64, activation="relu"))
model_cnn_4.add(Dense(units=38, activation="softmax"))

model_cnn_4.summary()

# Compilation du modèle cnn : model_cnn_4

model_cnn_4.compile(
    # loss = keras.losses.CategoricalFocalCrossentropy(),
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Entraînement du modèle cnn : model_cnn_4

epochs = 100
model_cnn_4_history = model_cnn_4.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint_freq, checkpoint_best, history_callback],
    verbose=True,
)

# Evaluation graphique de l'entraînement

# précisions d'entraînement et de validation obtenues pendant l'entraînement
train_acc_cnn_4 = model_cnn_4_history.history["accuracy"]
val_acc_cnn_4 = model_cnn_4_history.history["val_accuracy"]

# Courbe de la précision sur l'échantillon d'entrainement
plt.plot(
    np.arange(1, epochs + 1, 1),
    train_acc_cnn_4,
    label="CNN_4 Training Accuracy",
    color="red",
)

# Courbe de la précision sur l'échantillon de validation
plt.plot(
    np.arange(1, epochs + 1, 1),
    val_acc_cnn_4,
    label="CNN_4 Validation Accuracy",
    color="lightcoral",
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluation sur l'ensemble de test depuis un modèle chargé

# Chargement du modèle à évaluer dans classifier_cnn_4

model_path = "model_cnn_4_best.keras"  # chemin du modèle à charger

classifier_cnn_4 = keras.load_model(model_path)

# Scores

print("\n\nScores sur l'ensemble de test :", "\n\n")

scores_cnn_4 = classifier_cnn_4.evaluate(test_generator)

print("\n\nScores : \n\n", scores_cnn_4, end="\n\n")


# Predictions sur l'ensemble de test

# Prédictions sous forme de probas pour chaque classe
predictions_cnn_4_probas = classifier_cnn_4.predict(test_generator)

# Transformation des probabilités en classes prédites
predictions_cnn_4_classes = np.argmax(predictions_cnn_4_probas, axis=1)

# Classes issues de l'ensemble test
true_classes = test_generator.classes

# Labels des classes (= noms des dossiers)
class_labels = list(test_generator.class_indices.keys())


# Affichage de la matrice de confusion

cm = confusion_matrix(true_classes, predictions_cnn_4_classes)

print("\n\nPrédictions sur l'ensemble de test et matrice de confusion")
plt.figure(figsize=(15, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)

plt.title("Matrice de Confusion")
plt.ylabel("Classes Réelles")
plt.xlabel("Classes Prédites")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Affichage de la matrice de confusion normalisée

plt.figure(figsize=(20, 20))

cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)

plt.title("Matrice de Confusion normalisée")
plt.ylabel("Classes Réelles")
plt.xlabel("Classes Prédites")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()