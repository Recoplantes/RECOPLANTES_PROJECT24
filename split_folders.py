#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour diviser notre dataset en ensembles d'entraînement, validation et test.
Basé sur la librairie splitfolders.

Le répertoire "Data/plantvillage_dataset/color" contenant le dataset doit se 
trouver au même niveau que ce script.

Deux opérations sont réalisées :
1/  Split du dataset selon le ratio 0.8, 0.1, 0.1
    dans le répertoire de sortie "color_split"
2/  Split du dataset avec un nombre d'images fixes par ensemble 100, 25, 25
    dans le réperoire de sortie "color_split_light"
"""
# Librairies
import splitfolders

# Chemin vers le dataset à spliter
input_folder = "Data/plantvillage_dataset/color"

# Split du jeu de données avec un ratio de
# 80% pour l'ensemble d'entraînement
# 10% pour l'ensemble de validation
# 10% pour l'ensemble de test

splitfolders.ratio(
    input_folder,
    output="Data/color_split",
    seed=42,
    ratio=(0.8, 0.1, 0.1),
    group_prefix=None,
    move=False,  # les fichiers sont copiés et non déplacés
)


# Split du jeux de données avec un nombre fixes d'éléments par ensemble
# 100 images pour l'ensemble d'entraînement
# 25 images pour l'ensemble de validation
# 25 images pour l'ensemble de test

splitfolders.fixed(
    input_folder,
    output="Data/color_split_light",
    seed=42,
    fixed=(100, 25, 25),
    oversample=False,
    group_prefix=None,
    move=False,  # Les fichiers sont copiés et non déplacés
)
