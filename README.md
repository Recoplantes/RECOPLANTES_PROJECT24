Projet RECOPLANTS 2024

Nous souhaitons créer un projet de reconnaissance de plantes pour les professionnels. 
Le but est de créer une application avec un modèle d'apprentissage automatique qui, à partir d’une photo prise par l’utilisateur, reconnait le type de plante uniquement à partir de feuilles, et arrive à déterminer si la plante est malade et si oui de quelle maladie il s’agit. 
Notre application sera une application mobile qui permettra aux agriculteurs et agents de cultures de reconnaître les plantes malades dans leurs cultures. 
Cette application permettra aux professionnels de reconnaître ou prévenir les maladies afin d’adopter les bons gestes pour éviter l’utilisation de pesticide à haute quantité.

Dans le dataset situé à [/kaggle/input/plantvillage-dataset/color](https://www.kaggle.com/abdallahalidev/plantvillage-dataset),
il y a une banque d'images de feuilles saines ou présentant des symptômes de maladies ou d'attaques de ravageurs pour plusieurs espèces de plantes cultivées.
Chaque paire culture/maladie est enregistrée dans un dossier distinct. Pour chaque culture, il existe un répertoire intitulé xxx___healthy pour les feuilles saines. Les noms des dossiers peuvent donc servir de labels pour les données qu'ils contiennent, facilitant l'entraînement d'un modèle de classification supervisée.
Note : L'exploration se concentre sur le dossier "color", contenant les images originales, en excluant les autres dossiers (grayscale et segmented) qui contiennent les mêmes images déjà prétraitées

Liste des fichiers contenus dans ce dépôt :

model_cnn_4.py
Script pour construire, entraîner et évaluer le modèle: model_cnn_4 présenté dans notre rapport de modélisation

model_resnet_v7.py
Script pour construire, entraîner et évaluer le modèle model_resnet_v7 présenté dans notre rapport de modélisation

predictions_resnet_v7.py
Script pour réaliser des predictions à partir d'un modèle entraîné.
Preprocessing calibré pour réaliser des prédictions avec model_resnet_v7_best :
- fonction de peprocessing propre au modèle ResNet-50
- taille attendue des images en entrée : 224x224 pixels
