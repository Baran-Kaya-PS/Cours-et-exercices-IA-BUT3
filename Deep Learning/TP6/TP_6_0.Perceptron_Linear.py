#%% md
#  # TP6: Apprentissage profond(Deep learning)
#  ## TP6.0 Le perceptron from scratch: Régression linéaire/logistique  
#  Dans ce TP, nous explorerons le fonctionnement interne d'un neurone.
# 
#%% md
# #### Travail demandé :
# Objectif : démystifier la partie mathématiques dus réseaux de neurones
# - Définir un perceptron simple
# - Ecrire les fonctions qui permettet de faire tous les calculs 
# 
# On utilisera  deux datasets : le titanic et le housing. Un portant sur une classification l'aure sur une régression.
# Les programmes sont identifques seuls le calcul de l'activation, la loss et les dw et db 
# qui changent légérement. 
# <span style="color:green">  L'architecture du réseau à un réel impact sur les résultats </span>
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
#%% md
# ## Neurone/ Perceptron  
#%% md
# ### 1. Sélection des données 
#%% md
# ### Dataset pour la classification
#%%
# le dataset titanic 
import pandas as pd 
titanic = pd.read_csv('../../data/titanic.csv')

# On prend juste une partie du fichier.
titanic = titanic[['Survived', 'Pclass', 'Sex', 'Age']]
#  on regrade les chance de Survived  en fonction des autres features.
X_features=['Pclass', 'Sex', 'Age']
titanic.dropna(axis=0, inplace=True)
titanic['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
titanic.head()

y = titanic['Survived'] # récupérer la colonne survived et la mettre dans y

# récuperer le reste des données dans X utiliser la fonction titanic.drop ???, ??? )
X = titanic.drop('Survived', axis=1)
print(X.shape,y.shape)
#%% md
# ## 2. Préparation des données
#%% md
# #### Split des données 
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# reshape Y
y_train = y_train.values.reshape(-1,1) # -1 toutes les lignes, 1 une colonne
#%%
print(X_train.shape,y_train.shape)
#%% md
# ###   Normalisation des données
# Rappel : Il est primordial de normaliser les données. Vous pourrez vérifier les performances de vos modèles avec et sans données normalisées
#%%
## On peut utiliser une simple normalisation (X - moyenne)/ecart type)
# C'est une standardisation la même que StandardScaler de scikit learn
 
mean = X_train.mean()
std  = X_train.std()
X_train = (X_train - mean) / std

# Normalisation des données de test
X_test  = (X_test  - mean) / std

#display(X_train.describe().style.format("{0:.2f}").set_caption("After normalization :"))
display(X_train.head(3).style.format("{0:.2f}").set_caption("Few lines of the dataset :"))
#%% md
# ## 3. Construction (définition) du modèle 
# On construit un modèle de régression linéaire. 
# <figure>
#     <img src="../images/perceptron.png"  style="width:440px;height:200px;" >
# </figure>
# 
# $$ a=\sigma(z) = \frac{1}{1+\exp(-z)}$$
# 
# 
# 
#%% md
# ### 3.1. Fonctions du modele
# #### Petit rappel : on manipule directement les matrices et les vecteurs. Pas besoin de faire des boucles sur l'ensemble des valeurs, .... 
# #### Cette question a déjà été traitée dans un des TP précédents (vectorisation, voir lab_utils_multi.py TP2)
# 
# ### Questions : 
# Ecrire les fonctions ci-dessous
#%% md
# ### Initialisation des paramètres W et b
#%%
# Combien de paramètres 
def initialisation(X): # initialise les paramètres W et b
    W = np.random.randn(1,X.shape[1])   # X.shape[1] donne le nombre de paramètres, W = np.random.randn(1, X.shape[1]) car on a un seul neurone et X.shape[1] paramètres
    b = np.random.randn(1) # b = np.random.randn(1) car on a un seul neurone
    return (W, b)
#%% md
# ### Calcul activation du modèle (Z et A)
#%%
# Calcul activation du modèle Z et A 
def model(X, W, b): # X représente les données, W et b les paramètres du modèle
    Z =  np.dot(X,W.T)+b# Z est égal a X*W+b
    Z = np.clip(Z, -500, 500) # pour éviter les erreurs de calculs, clip sert a limiter les valeurs de Z entre -500 et 500
    A = 1/(1+np.exp(-Z))# A est égal à la sigmoide de Z, la fonction sigmoide = 1 divisé par 1 + exp(-Z), Sigmoide = 1/(1+np.exp(-Z))
    return A
#%% md
# ### Calcul de la Loss
# $$loss=1/m * \sum (y * log(A) - (1 - y) * log(1 - A))$$
#%%
# Calcler la Loss (la MSE
def log_loss(A, y):  # A est l'activation du modèle, y les vraies valeurs
    # Il faut calculer la loss, voici la fonction : loss=1/m * \sum (y * log(A) - (1 - y) * log(1 - A))
    # m = y.shape[0] # nombre d'exemples
    # sum (y * log(a) représente la somme des y * log(a) pour chaque exemple
    return 1/y.shape[0] * y*np.log(A) - (1-y) * np.log(1-A)
    
#%% md
# ### Calcul des gradients
#%%
def gradients(A, X, y):
    # pour calculer le gradient il faut : dw qui est égal a 1/m * X * (A - y) et db qui est égal a 1/m * (A - y)
    # m = y.shape[0] # nombre d'exemples
    dW = np.dot(X.T, (A-y) / y.shape[0])
    db = np.sum(A-y) / y.shape[0]
    return dW, db
#%% md
# ### Modifier w et b
#%%
def update(dW, db, W, b, learning_rate): # learning_rate est le taux d'apprentissage
    W = W - learning_rate*dW # W représente les paramètres du modèle et est égal à W - learning_rate * dW
    b = b-learning_rate*db # b représente les paramètres du modèle et est égal à b - learning_rate * db
    return (W, b)
#%% md
# ### Prediction 
#%%
def predict(X, W, b): # prédict permet de prédire quelque chose en fonction de nouvelles données qui ne sont pas dans le dataset
    A = model(X, W, b)
    return (A >= 0.5).astype(int).reshape(-1,1) # classification, retourne vrai si A est supérieur ou égal à 0.5, faux sinon
    
#%% md
# ### Définition du neurone et de ses fonctions : mettre tout dans une même fonction
# - initialiser les paramètres du modèle
# - définir les hyperparamètres : learning_rate, nombre_d'itérations
# - commencer les traitements (dans une boucle) 
#     - appel da la fonction qui calcule l'activation
#     - caculer l'erreur (pour la visualiser) 
#     - calculer les gradients
#     - effectuer les modification des paramètres w et b
#     - réitérer
# - une fois le modèle appris
#     - commencer la prédiction
#     - visualiser la Loss et les performances
# 
#%%
def perceptron(X, y, learning_rate, n_iter):
    # initialisation W, b
    W, b = initialisation(X)
    Loss = []
    Les_W =[]
    # itérer l'apprentissage 
    for i in range(n_iter):
        #Calculer l'activation du neurone
        A = model(X,W,b) # appel de la fonction model A = model(X, W, b)
        #Calculer la Loss mettre dans une liste
        Loss.append(log_loss(A, y))
        # Calul des gradients
        dW, db = gradients(A, X, y) # appel de la fonction gradients dW, db = gradients(A, X, y)
        
        # Modifier les paramètres
        W, b = update(dW, db, W, b, learning_rate) # appel de la fonction update W, b = update(dW, db, W, b, learning_rate
        
    return (W, b, Loss)
    
#%%
learning_rate = 0.5
n_iter = 10000
W, b , Loss = perceptron(X_train, y_train, learning_rate,n_iter )
#%% md
# #### Calcul des prédictions et les performances appropriées (Accuracy ou r2_score)
#%%
# Prédictions pour l'ensemble d'entraînement
y_pred_train = predict(X_train, W, b)  # Utilisez la fonction predict pour obtenir les prédictions

# Prédictions pour l'ensemble de test
y_pred = predict(X_test, W, b)  # Utilisez la fonction predict ici aussi

# Afficher les formes pour vérifier la consistance
print(y_train.shape, y_pred_train.shape)
print(y_test.shape, y_pred.shape)

# Calcul de l'accuracy
print("Accuracy train", accuracy_score(y_train, y_pred_train), "Accuracy Test", accuracy_score(y_test, y_pred))

#%% md
# ### Visualiser la Loss
#%%
plt.plot(Loss)
plt.show()
#%% md
# ## Questions
# L'objectif dec cette partie est d'évaluer l'impact du taux d'apprentissage (learning_rate). 
# - 1. Reprendre l'algorithme d'apprentissage, mettre en place un learning_rate dynamique, qui est mis à jour chaque fois que l'iteration, $t$ (qui correspond à l'indice $i$ dans le programme principal) prend une valeur multiple de $10$, $100$, vous pourrez utiliser la forme suivante : $\alpha=\frac{\alpha}{\sqrt(t+1)}$, t étant l'itération. 
# - 2. Comparer les courbes des Loss obtenues pour chacun de ces cas (learning_rate constant et learning_rate dynamique avec $t$ aleur multiple de $10$, $100$).
# - 3. On souhaite arrêter l'apprentissage soit quand l'erreur est inférieure à 0.1 ou quand on atteint le nombre diitérations max, rajouter cette contrainte à l'algortime   
# - 4. Refaire l'exercice en prenant les données du fichier smoking (du contôle). Le dataset est dans la cellule ci dessous. 
# 
#%%
# Lecture du dataset n permet de limiter le nombre de lignes à lire
# Pour faciliter les tests
import pandas as pd 
def load_data(n):
    data = pd.read_csv('../../data/train.csv')
    return data[0:n]

# le -1 du load_data(-1) veut dire on prend toutes les lignes 
data=load_data(-1)
data.dropna(axis=0, inplace=True)
y = data['smoking'] # récupérer la colonne survived et la mettre dans y
# récuperer le reste des données dans X utiliser la fonction titanic.drop ???, ??? )
X = data.drop('smoking', axis=1)
#%%
