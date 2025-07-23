# MNIST Neural Network from Scratch

Une implémentation complète d'un réseau de neurones pour la reconnaissance de chiffres manuscrits, développée entièrement from scratch avec NumPy.

## Objectif du projet

Ce projet démontre une compréhension profonde des réseaux de neurones en implémentant :
- La rétropropagation (backpropagation)
- La descente de gradient (gradient descent)
- Les fonctions d'activation
- L'entraînement et l'évaluation d'un modèle

## Dataset

Le dataset MNIST contient 70,000 images de chiffres manuscrits (0-9) :
- 60,000 images d'entraînement
- 10,000 images de test
- Images en niveaux de gris 28x28 pixels

lien : https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## Ressources

- [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (Vidéo 1 à 4)
- [Guy that does exactly this exercise](https://www.youtube.com/watch?v=w8yWXqWQYmU).
- [Article about this exercise](https://medium.com/@ombaval/building-a-simple-neural-network-from-scratch-for-mnist-digit-recognition-without-using-7005a7733418)
- [Easy explanation to gradient descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Guy that builds a small GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Technologies utilisées

- **Python 3.8+**
- **NumPy** - Calculs mathématiques
- **Matplotlib** - Visualisations
- **Scikit-learn** - Chargement des données et comparaison

## Structure du projet

```
mnist-neural-network-from-scratch/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sklearn_baseline.ipynb
│   └── 03_from_scratch_neural_net.ipynb
└── results/
    └── ... (figures, courbes, etc.)
```

## Comment utiliser ce projet

### Installation
```bash
git clone https://github.com/elsanto131/mnist-neural-network-from-scratch
cd mnist-neural-network-from-scratch
pip install -r requirements.txt
```

## Résultats

- Précision obtenue sur le jeu de test : **97.12%** (à compléter)
- Visualisation des erreurs et des exemples mal classés
- Diminution régulière de la loss pendant l’entraînement

## Ce que j'ai appris

- Comment fonctionne la propagation avant et la rétropropagation
- L’importance des dérivées et du calcul des gradients
- Pourquoi on normalise les données d’entrée
- Les limites d’un réseau simple et l’intérêt des frameworks spécialisés

## Remerciements

Ce projet a été réalisé dans le cadre de ma formation **AI Bootcamp** chez **BeCode**.

Un grand merci à :
- **BeCode** pour cette formation exceptionnelle en intelligence artificielle
- **Mon coach Antoine AI Coach** pour son accompagnement et ses conseils précieux
- **Ma promotion LGG-Thomas5** pour l'entraide et la motivation collective : Alex, Mai-ly, Nat, Julie, Quentin One, Natalya, Robin, Gaetan, Raf, Hervé, Marty, Riccardo, Quentin B., Olesia, Elsa, Hang, Miao, Konstantin, Arvind, Waseem et Cindy.
- **La communauté BeCode** pour cet environnement d'apprentissage stimulant : Antoine, Loic, Benja, Medhi


Cette formation m'a permis de passer de zéro connaissance en Python à la création d'un réseau de neurones from scratch en quelques mois !

## Auteur

**Santo D'Acquisto** - www.linkedin.com/in/s-dacquisto - elsanto131@hotmail.com