
# Ce script utilise MobileNetV2, un modèle préentraîné sur ImageNet, adapté
# pour une tâche de classification d'images de fruits et légumes. MobileNetV2
# est particulièrement efficace pour des environnements contraints en
# ressources grâce à son architecture optimisée pour la vitesse et la légèreté.
# En complément, une régularisation L2 est intégrée dans la dernière couche dense
# pour limiter le surapprentissage en pénalisant les poids excessifs, garantissant
# ainsi une meilleure généralisation du modèle.
# 
# Dans ce projet, les principales étapes incluent :
# 1. Préparation des données avec augmentation pour améliorer la robustesse
#    et réduire les biais liés aux variations des images.
# 2. Utilisation de MobileNetV2 pour extraire des caractéristiques visuelles
#    complexes à partir des images redimensionnées à 96x96 pixels.
# 3. Ajout de couches personnalisées pour la classification des classes
#    correspondantes aux fruits et légumes dans le dataset.
# 4. Entraînement progressif par blocs de 5 époques pour surveiller les
#    performances et prévenir le surapprentissage.
# 5. Sauvegarde des résultats intermédiaires et combinaison des historiques
#    pour finaliser l'entraînement avec 25 époques au total.
# 
# Chaque étape est soigneusement configurée pour tirer parti des capacités
# de MobileNetV2 et de la régularisation L2, tout en respectant les contraintes
# des ressources disponibles dans un environnement comme Google Colab.

# Pendant l'entraînement, le script était initialement conçu pour exécuter 20 époques
# en blocs de 5. Cependant, au cours de l'exécution, des anomalies ont été observées :
# - Bloc 1 : 5 époques ont été exécutées comme prévu.
# - Bloc 2 : 10 époques ont été réalisées au lieu de 5.
# - Bloc 3 : En cours pour 15 époques, j'ai décidé d'interrompre manuellement
#   l'entraînement à la 10ème époque (soit la 25ème époque au total).
# Cela a permis de limiter le risque de surapprentissage. Après l'entraînement,
# un script a été utilisé pour sauvegarder manuellement l'historique du Bloc 3,
# le combiner avec les blocs précédents, et poursuivre l'analyse sur l'ensemble de test.

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
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import datetime
import json
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix

# DÉFINITION DES CHEMINS DACCÈS AUX DONNÉES
train_path = './data/train'
val_path = './data/val'
test_path = './data/test'
mob_path = './saved_models/'

print("Train path exists:", os.path.exists(train_path))
print("Validation path exists:", os.path.exists(val_path))
print("Test path exists:", os.path.exists(test_path))
if not os.path.exists(mob_path):
    # Crée le répertoire local pour sauvegarder les modèles et historiques
    os.makedirs(mob_path)

# PRÉPARATION DES GÉNÉRATEURS D'IMAGES
train_data_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_data_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_data_generator.flow_from_directory(
    train_path,
    target_size=(96, 96),
    batch_size=16,
    class_mode='categorical'
)

val_generator = val_test_data_generator.flow_from_directory(
    val_path,
    target_size=(96, 96),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)
test_generator = val_test_data_generator.flow_from_directory(
    test_path,
    target_size=(96, 96),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# CONFIGURATION DES CALLBACKS
checkpoint_best = ModelCheckpoint(
    filepath=f'{mob_path}Anas_Essai_1_MOB_L2.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# CONSTRUCTION DU MODÈLE AVEC MOBILENETV2 AVEC L2
# Initialisation du modèle MobileNetV2 préentraîné avec réglage des couches givrées.
# Charger MobileNetV2 préentraîné
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(96, 96, 3)
)
base_model.trainable = True

# Geler les couches sauf les 20 dernières pour le fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = keras.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(
    train_generator.num_classes, activation='softmax', kernel_regularizer=l2(0.01)
)(x)
model = keras.Model(inputs, outputs)

# Configurer le modèle avec l'optimiseur Adam, la perte catégorielle et les métriques
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ENTRAÎNEMENT DU MODÈLE PAR BLOCS
# Formation progressive avec des blocs pour limiter les risques de surapprentissage.
block_size = 5
initial_epochs = 20
current_date = datetime.datetime.now().strftime("%Y%m%d")

# Boucle pour entraîner le modèle par blocs progressifs
for i in range(0, initial_epochs, block_size):
    history_block = model.fit(
        train_generator,
        epochs=min(i + block_size, initial_epochs),
        validation_data=val_generator,
        callbacks=[checkpoint_best, early_stopping],
        verbose=1
    )
    with open(f'{mob_path}history_block_{i + 1}_{current_date}.json', 'w') as f:
        json.dump(history_block.history, f)
    model.save(f'{mob_path}model_l2_after_block_{i + 1}_{current_date}.keras')

# Chemin vers l'historique partiel du Bloc 3
bloc3_partial_path = f'{mob_path}manual_history_block3_1_to_8_{current_date}.json'

with open(bloc3_partial_path, 'r') as f:
    manual_block_3 = json.load(f)

print("Clés dans l'historique du Bloc 3 :", manual_block_3.keys())
print("Nombre d'époques enregistrées :", len(manual_block_3['accuracy']))

bloc1_path = './saved_models/history_block_1_20241208.json'
bloc2_path = './saved_models/history_block_6_20241208.json'
bloc3_partial_path = './saved_models/manual_history_block3_1_to_8_20241208.json'

with open(bloc1_path, 'r') as f:
    history_block_1 = json.load(f)

with open(bloc2_path, 'r') as f:
    history_block_2 = json.load(f)

with open(bloc3_partial_path, 'r') as f:
    manual_block_3 = json.load(f)

print("Bloc 1 : ", len(history_block_1['accuracy']))
print("Bloc 2 : ", len(history_block_2['accuracy']))
print("Bloc 3 : ", len(manual_block_3['accuracy']))

# Combiner les historiques des trois blocs
combined_history = {
    key: history_block_1[key] + history_block_2[key] + manual_block_3[key]
    for key in history_block_1.keys()
}

combined_path = f'{mob_path}combined_history_25_epochs_{current_date}.json'
with open(combined_path, 'w') as f:
    json.dump(combined_history, f)

print(f"L'historique global (25 époques) a été sauvegardé dans {combined_path}.")

# Tracer les courbes d'apprentissage
plt.plot(combined_history['accuracy'], label='Train Accuracy')
plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy par époque')
plt.show()

plt.plot(combined_history['loss'], label='Train Loss')
plt.plot(combined_history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss par époque')
plt.show()

# PRÉDICTIONS ET MATRICE DE CONFUSION
# Évaluer les performances sur les données de test
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Gérer les prédictions et la matrice de confusion
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Générer la matrice de confusion normalisée
cm = confusion_matrix(true_classes, predicted_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Configurer la taille et les limites d'affichage
plt.figure(figsize=(18, 16))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=45, fontsize=8, ha='right')
plt.yticks(fontsize=8)
plt.title("Matrice de Confusion Normalisée - MobileNetV2", fontsize=16)
plt.xlabel("Classes Prédites", fontsize=12)
plt.ylabel("Classes Réelles", fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_improved_v3.png', dpi=300)
plt.show()
