# MNIST Neural Network from Scratch | Réseau de Neurones MNIST From Scratch

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/NumPy-1.21+-green.svg" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-3.5+-orange.svg" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0+-red.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Accuracy-97.12%25-brightgreen.svg" alt="Accuracy">
</div>

<div align="center">
  <a href="#english">🇺🇸 English</a> | <a href="#français">🇫🇷 Français</a>
</div>

---

## English

A complete neural network implementation for MNIST handwritten digit recognition, developed entirely **from scratch** using NumPy.

### Project Objective

This project demonstrates a deep understanding of neural networks by implementing all essential components:

- **Forward Propagation**
- **Backpropagation** 
- **Gradient Descent**
- **Activation Functions** (ReLU, Softmax)
- **Mini-batch Training**
- **Results Evaluation and Visualization**
- **Binary File Manipulation** (.idx format)

### Dataset

**MNIST** - Modified National Institute of Standards and Technology database
- **70,000 images** of handwritten digits (0-9)
- **60,000 training images**  
- **10,000 test images**
- **Format**: 28x28 pixels in grayscale

**Source**: [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

### Network Architecture

```
Input (784) → Hidden Layer (64) → Output (10)
     |              |                  |
   Pixels       ReLU Activation    Softmax
  28x28=784      64 neurons       10 classes
```

- **Input Layer**: 784 neurons (one per pixel)
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

### Installation and Usage

#### Prerequisites
```bash
Python 3.8+
NumPy
Matplotlib
Scikit-learn
```

#### Installation
```bash
git clone https://github.com/elsanto131/mnist-neural-network-from-scratch
cd mnist-neural-network-from-scratch
pip install -r requirements.txt
```

#### Execution
1. **Data Exploration**: `notebooks/01_data_exploration.ipynb`
2. **Scikit-learn Baseline**: `notebooks/02_sklearn_baseline.ipynb`  
3. **From Scratch Network**: `notebooks/03_from_scratch_neural_net.ipynb`

### Results

| Model | Accuracy | Training Time | Comment |
|-------|----------|---------------|---------|
| **Scikit-learn Baseline** | 92.03% | ~30s | Logistic Regression |
| **Neural Network from Scratch** | **97.12%** | ~2min | 10 epochs, batch_size=64 |

### Implemented Concepts

#### Mathematical Functions
- **ReLU**: `f(x) = max(0, x)`
- **Softmax**: `f(xi) = exp(xi) / Σ exp(xj)`
- **Cross-Entropy Loss**: `L = -Σ y_true * log(y_pred)`

#### Algorithms
- **Forward Propagation**: Prediction calculation
- **Backpropagation**: Gradient calculation via chain rule
- **Gradient Descent**: Weight update `W = W - η * ∇W`

### What I Learned

- **Deep understanding** of forward and backpropagation
- **Critical importance** of derivatives and gradient calculation
- **Need to normalize** input data (0-255 → 0-1)
- **Binary file manipulation** and .idx data structure
- **Balance** between model simplicity and performance
- **Limitations** of simple networks vs specialized frameworks

---

## Français

Une implémentation complète d'un réseau de neurones pour la reconnaissance de chiffres manuscrits MNIST, développée entièrement **from scratch** avec NumPy.

### Objectif du projet

Ce projet démontre une compréhension profonde des réseaux de neurones en implémentant tous les composants essentiels :

- **Propagation avant** (Forward Propagation)
- **Rétropropagation** (Backpropagation) 
- **Descente de gradient** (Gradient Descent)
- **Fonctions d'activation** (ReLU, Softmax)
- **Entraînement par batch** (Mini-batch Training)
- **Évaluation et visualisation** des résultats
- **Manipulation de fichiers binaires** (format .idx)

### Dataset

**MNIST** - Modified National Institute of Standards and Technology database
- **70,000 images** de chiffres manuscrits (0-9)
- **60,000 images** d'entraînement  
- **10,000 images** de test
- **Format** : 28x28 pixels en niveaux de gris

**Source** : [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

### Architecture du réseau

```
Entrée (784) → Couche cachée (64) → Sortie (10)
     |              |                  |
   Pixels       ReLU Activation    Softmax
  28x28=784      64 neurones      10 classes
```

- **Couche d'entrée** : 784 neurones (un par pixel)
- **Couche cachée** : 64 neurones avec activation ReLU
- **Couche de sortie** : 10 neurones avec activation Softmax

### Installation et utilisation

#### Prérequis
```bash
Python 3.8+
NumPy
Matplotlib
Scikit-learn
```

#### Installation
```bash
git clone https://github.com/elsanto131/mnist-neural-network-from-scratch
cd mnist-neural-network-from-scratch
pip install -r requirements.txt
```

#### Exécution
1. **Exploration des données** : `notebooks/01_data_exploration.ipynb`
2. **Baseline Scikit-learn** : `notebooks/02_sklearn_baseline.ipynb`  
3. **Réseau from scratch** : `notebooks/03_from_scratch_neural_net.ipynb`

### Résultats

| Modèle | Précision | Temps d'entraînement | Commentaire |
|--------|-----------|---------------------|-------------|
| **Scikit-learn Baseline** | 92.03% | ~30s | Régression logistique |
| **Neural Network from Scratch** | **97.12%** | ~2min | 10 epochs, batch_size=64 |

### Concepts implémentés

#### Fonctions mathématiques
- **ReLU** : `f(x) = max(0, x)`
- **Softmax** : `f(xi) = exp(xi) / Σ exp(xj)`
- **Cross-Entropy Loss** : `L = -Σ y_true * log(y_pred)`

#### Algorithmes
- **Forward Propagation** : Calcul des prédictions
- **Backpropagation** : Calcul des gradients via la règle de dérivation en chaîne
- **Gradient Descent** : Mise à jour des poids `W = W - η * ∇W`

### Ce que j'ai appris

- **Compréhension profonde** de la propagation avant et de la rétropropagation
- **Importance cruciale** des dérivées et du calcul des gradients
- **Nécessité de normaliser** les données d'entrée (0-255 → 0-1)
- **Manipulation de fichiers binaires** et structure des données .idx
- **Équilibre** entre simplicité et performance d'un modèle
- **Limites** d'un réseau simple vs frameworks spécialisés

---

### Project Structure | Structure du projet

```
mnist-neural-network-from-scratch/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── train-images.idx3-ubyte
│       ├── train-labels.idx1-ubyte
│       ├── t10k-images.idx3-ubyte
│       └── t10k-labels.idx1-ubyte
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sklearn_baseline.ipynb
│   └── 03_from_scratch_neural_net.ipynb
├── picture/
│   └── reseau_de_neurones_simple_opeclassroom.jpg
└── results/
    └── ... (graphs, metrics, etc.)
```

### Resources and Inspiration | Ressources et inspiration

- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (Videos 1-4)
- [Neural Network from Scratch Tutorial](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [Medium Article - MNIST from Scratch](https://medium.com/@ombaval/building-a-simple-neural-network-from-scratch-for-mnist-digit-recognition-without-using-7005a7733418)
- [Gradient Descent Explained](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Official MNIST Documentation](http://yann.lecun.com/exdb/mnist/)

### Acknowledgments | Remerciements

This project was completed as part of my **AI Bootcamp** training at **BeCode**.  
*Ce projet a été réalisé dans le cadre de ma formation **AI Bootcamp** chez **BeCode**.*

**A big thank you to | Un grand merci à :**
- **BeCode** for this exceptional artificial intelligence training | pour cette formation exceptionnelle en intelligence artificielle
- **Antoine (AI Coach)** for his guidance and valuable advice | pour son accompagnement et ses conseils précieux
- **My LGG-Thomas5 cohort** for mutual support and collective motivation | **Ma promotion LGG-Thomas5** pour l'entraide et la motivation collective : Alex, Mai-ly, Nat, Julie, Quentin One, Natalya, Robin, Gaetan, Raf, Hervé, Marty, Riccardo, Quentin B., Olesia, Elsa, Hang, Miao, Konstantin, Arvind, Waseem et Cindy.
- **The BeCode community** for this stimulating learning environment | **La communauté BeCode** pour cet environnement d'apprentissage stimulant : Antoine, Loic, Benja, Medhi

*This training allowed me to go from zero Python knowledge to creating a neural network from scratch in just a few months!*  
*Cette formation m'a permis de passer de zéro connaissance en Python à la création d'un réseau de neurones from scratch en quelques mois !*

### Author | Auteur

**Santo D'Acquisto**  
 [LinkedIn](https://www.linkedin.com/in/s-dacquisto)  
[GitHub](https://github.com/elsanto131)

---
<div align="center">
  <i>Made with heart and lots of coffee | Fait avec coeur et beaucoup de café</i>
</div>