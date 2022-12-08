#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:34:02 2021

@author: dejan
"""
from visualize import ShowSelected
from read_WDF import read_WDF


# Nom de fichier avec le chemin pour y accèder:
filename = "data/exampleA1.wdf"
# charger le fichier:
da, img = read_WDF(filename)
# %% Visualiser:
check = ShowSelected(da, epsilon=2)
# %% Pour récupérer les valeurs affichées:
valeurs = check.imup.get_array()
