
"""
Ce script utilise MobileNetV2, un modèle préentraîné sur ImageNet, adapté
pour une tâche de classification d'images personnalisée. MobileNetV2 est
particulièrement efficace pour des environnements contraints en ressources
grès à son architecture optimisée pour la vitesse et la légèreté.

Dans ce projet, les principales étapes incluent :
1. Préparation des données avec augmentation pour améliorer la robustesse.
2. Utilisation de MobileNetV2 pour extraire les caractéristiques des images.
3. Ajout de couches personnalisées pour la classification.
4. Entraînement progressif par blocs pour réduire le surapprentissage.
5. Fine-tuning du modèle pour améliorer les performances.

Chaque étape est soigneusement configurée pour optimiser les performances
tout en respectant les contraintes des ressources disponibles.
"""

# IMPORTATION DES LIBRAIRIES
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json
from tensorflow.keras.applications import MobileNetV2

# DÉFINITION DES CHEMINS D'ACCÈS AUX DONNÉES
# Chemins vers les ensembles d'entraînement, validation et test
data_train_path = './data/color_split/train'
data_val_path = './data/color_split/val'
data_test_path = './data/color_split/test'

# Création des répertoires nécessaires
os.makedirs('./models', exist_ok=True)

# PRÉPARATION DES GÉNÉRATEURS D'IMAGES
# Définition des paramètres pour les générateurs d'images
batch_size = 16
img_height = 96
img_width = 96

# Générateur pour les données d'entraînement avec augmentation de données
train_data_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Générateur pour les données de validation et de test sans augmentation
val_test_data_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Création des générateurs pour les ensembles d'entraînement, validation et test
train_generator = train_data_generator.flow_from_directory(
    directory=data_train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_data_generator.flow_from_directory(
    directory=data_val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_data_generator.flow_from_directory(
    directory=data_test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# CONFIGURATION DES CALLBACKS
current_date = datetime.datetime.now().strftime("%Y%m%d")  # Date courante formatée pour les fichiers

checkpoint_best = ModelCheckpoint(  # Callback pour sauvegarder le meilleur modèle
    filepath=f'./models/Anas_Essai_1_MOB_Repeat.keras',  # Chemin du fichier sauvegardé
    monitor='val_accuracy',  # Surveiller la précision de validation
    save_best_only=True,  # Sauvegarder uniquement le meilleur modèle
    mode='max',  # Maximiser la métrique surveillée
    verbose=1  # Afficher des messages pendant la sauvegarde
)

early_stopping = EarlyStopping(  # Callback pour arrêter l'entraînement si nécessaire
    monitor='val_loss',  # Surveiller la perte de validation
    patience=5,  # Nombre d'époques sans amélioration avant l'arrêt
    verbose=1,  # Afficher des messages pendant l'arrêt
    restore_best_weights=True  # Restaurer les poids du meilleur modèle
)

# CONSTRUCTION DU MODÈLE AVEC MOBILENETV2
# Initialisation du modèle préentraîné MobileNetV2 sans les couches supérieures
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
# On gèle les poids du modèle de base pour éviter qu'ils ne soient modifiés
base_model.trainable = False

# Ajout des couches personnalisées
# Ces couches permettent d'adapter le modèle préentraîné à notre tâche
inputs = keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)  # Extraction de caractéristiques
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Réduction dimensionnelle
x = Dropout(0.2)(x)  # Réduction du surapprentissage
outputs = Dense(train_generator.num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compilation du modèle
# Configuration de l'optimiseur, de la fonction de perte et des métriques
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Résumé du modèle
# Affiche la structure du modèle et les paramètres
model.summary()

# ENTRAÎNEMENT DU MODÈLE PAR BLOCS
# On entraîne le modèle par blocs pour éviter un surapprentissage
block_size = 5
initial_epochs = 20

for i in range(0, initial_epochs, block_size):
    print(f"Bloc d'époques {i + 1} à {min(i + block_size, initial_epochs)}")
    history_block = model.fit(
        train_generator,
        epochs=min(i + block_size, initial_epochs),
        validation_data=val_generator,
        callbacks=[
            checkpoint_best,
            early_stopping
        ],
        verbose=1
    )

    # Sauvegarde de l'historique d'entraînement par bloc
    with open(f'./models/history_block_{i + 1}_{current_date}.json', 'w') as f:
        json.dump(history_block.history, f)

    # Sauvegarde des poids après chaque bloc
    model.save(f'./models/model_after_block_{i + 1}_{current_date}.keras')

# ÉVALUATION DU MODÈLE ET AFFICHAGE DES RESULTATS
# Évaluation finale du modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Loss: {test_loss}, Accuracy: {test_accuracy}')

# PRÉDICTIONS ET RAPPORT DE CLASSIFICATION
# Calcul des prédictions et comparaison avec les classes réelles
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # Classes prédites
true_classes = test_generator.classes  # Classes réelles
class_labels = list(test_generator.class_indices.keys())

# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(true_classes, predicted_classes)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 8))
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

# FINE-TUNING DU MODÈLE
# Déblocage des dernières couches du modèle pour améliorer ses performances
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False  # On gèle toutes les couches sauf les 30 dernières

# Recompilation du modèle avec un taux d'apprentissage réduit
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Taux d'apprentissage faible
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
# Entraînement des couches débloquées
model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    verbose=1
)

# Évaluation après fine-tuning
# Évaluation finale pour vérifier l'amélioration après fine-tuning
test_loss_ft, test_accuracy_ft = model.evaluate(test_generator)
print(f'Loss après fine-tuning: {test_loss_ft}, Accuracy après fine-tuning: {test_accuracy_ft}')

# TRACER LES COURBES D'APPRENTISSAGE
# Visualisation des performances pendant l'entraînement
history = history_block.history
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Précision Entraînement')
plt.plot(history['val_accuracy'], label='Précision Validation')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()
plt.title('Courbes de Précision')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Perte Entraînement')
plt.plot(history['val_loss'], label='Perte Validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.title('Courbes de Perte')
plt.show()
