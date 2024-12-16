### Projet de Reconnaissance de Plantes avec TensorFlow et Keras

## **Streamlit**
- Lien : https://reco-plantes.streamlit.app/

## **Description du projet**

Ce projet utilise **TensorFlow** et **Keras** pour construire un modèle de reconnaissance d'images de plantes basé sur un réseau de neurones convolutifs (CNN). L'objectif principal est de former un modèle capable de reconnaître et de classer des images en fonction de certaines caractéristiques. Ce projet implique la manipulation d'images, le prétraitement des données et la construction d'un modèle de classification.

**Données source :**  
- **Kaggle**  
- **PlantVillage Dataset** : Dataset d'images de feuilles de plantes malades et saines, avec les étiquettes correspondantes.  
- Lien : [PlantVillage Dataset](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)

---

## **Prérequis**

Avant de commencer, vous devez vous assurer que vous avez installé **Python 3.x** et les bibliothèques suivantes :

- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- PIL (Pillow)

---

## **Installation**

1. **Clonez ce projet dans votre répertoire local :**
   ```bash
   git clone https://github.com/AnasMba19/Reco-Plantes.git
   ```

2. **Créez un environnement virtuel :**
   ```bash
   python -m venv env
   ```

3. **Activez votre environnement virtuel :**
   - Windows : `\.\env\Scripts\activate`
   - Mac/Linux : `source env/bin/activate`

4. **Installez les dépendances nécessaires :**
   ```bash
   pip install -r requirements.txt
   ```

5. **Préparez vos données d'images :**
   Placez vos images dans le dossier approprié (`test_images/`) pour effectuer des prédictions.

6. **Exécutez l'application Streamlit pour visualiser les prédictions :**
   ```bash
   streamlit run app.py
   ```

---

## **Modèles utilisés**

### **1. Modèle MobileNetV2**

#### **Essai 1**
- Entraînement avec régularisation L2 pour limiter le surapprentissage.  
- Nombre d'epochs : **25**.  

#### **Essai 2**
- Entraînement en deux phases :  
  - **20 epochs** pour l'entraînement initial.  
  - **10 epochs** supplémentaires pour un fine-tuning précis.

---

### **2. Modèle TEM3**
- Prétraitement des images avec **ImageDataGenerator** de Keras.
- Utilisation de **ModelCheckpoint** pour sauvegarder le meilleur modèle.
- Formation avec compatibilité assurée pour **TensorFlow**.

---

### **3. Modèle ResNet_v7**
- Construit à l'aide de l'architecture ResNet50.  
- Taille des images en entrée : **224x224 pixels**.  
- Prétraitement spécifique au modèle ResNet.

---

## **Exécution du projet**

Pour effectuer des prédictions sur des images :

1. Lancez l'application Streamlit avec `app.py`.  
2. Sélectionnez un modèle parmi les options disponibles.  
3. Chargez une image depuis le dossier `test_images/`.  
4. Obtenez la prédiction affichée dans l'interface utilisateur avec la classe et la confiance associées.

---

## **Structure du projet**

- `models/` : Contient les fichiers `.keras` pour les modèles préentrainés.
- `test_images/` : Dossier pour tester des prédictions avec des images.
- `app.py` : Script principal pour exécuter l'application Streamlit.
- `requirements.txt` : Liste des dépendances nécessaires.

---

