#Analyse-des-expressions-faciales

modeles: CNN et SVM

#Introduction
Ce projet vise à classer l'émotion sur le visage d'une personne dans l'une des sept catégories, en utilisant des réseaux neuronaux convolutionnels profonds et les Machines à Vecteurs de Support. Les modèle sont formés sur l'ensemble de données provenant du kaggle: https://www.kaggle.com/jonathanoheix/face-expression-recognition- dataset. Cet ensemble de données se compose de images de visages en niveaux de gris divisé pour le test et lentrainement, de taille 48x48, avec sept émotions en colère, dégoûté, craintif, heureux, neutre, triste et surpris.
#Structue du dossier pour CNN:

CNN F.py
haarcascade frontalface_default(1).xml (file)
model(1).h5 (file)


#NOTE
La méthode de la cascade de Haar est utilisée pour détecter les visages dans chaque image de la diffusion de la webcam.
