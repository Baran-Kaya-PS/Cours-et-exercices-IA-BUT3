#%% md
#  # TP6: Apprentissage profond(Deep learning)
#  ## TP6.2 Un modèle multicouches pour une Régression  
#  On reprend le TP6.1 on change de modèle on prend un modèle multicouches. 
# 
#%% md
# #### Travail demandé :
# Objectif  comparer les performances 
# - d'un réseau de neurones simple (un Perceptrion, 1 seul neurone de sortie) (TP 6.1) 
# - et un réseau plus dense avec une couche cachée et une couche de sortie composée d'un seul neurone. (TP6.2)
# 
# De même nous travaillerons sur deux datasets. Les deux datasets portent sur le housing. On vous propose deux Datasets : 1 (small) à 4 variables et 100 exemples et le second à 13 variables et quelques centaines d'exemples.
# <span style="color:green">  L'architecture du réseau à un réel impact sur les résultats </span>
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split

#%%
#fonction utile pour calculer le R2 score
from keras import backend as K

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
#%% md
# ## Neurone sans activation - Régression linéaire
#%% md
# ### 1. Sélection des données 
# On utilisera les données houses.
# On comence tout d'abord par le petit  (Small) Dataset puis on refait le process en prenant le Large dataset.
#%% md
# #### Small dataset
#%%
# lecture du fichier texte.
from sklearn.model_selection import train_test_split
data=pd.read_csv("../../data/houses.txt", header=None)
data = data.rename(columns={0: 'Surf', 1: 'Nbpieces', 2: 'nbEtage', 3: 'Age', 4:'Prix'})
X= data.drop('Prix',  axis=1)
y=data['Prix']


#%%
## affichage des noms de colonnes (variables du modèle) 
X.columns
#%% md
# ## 2 Préparation des données
#%% md
# #### Split des données 
#%%
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
print(X.shape, y.shape, X_test.shape, y_test.shape)
#%% md
# #### 2.1  Normalisation des données
#%%
## On peut utiliser une simple normalisation (x-mu)/ecart type)
# C'est une standardisation la même que StandardScaler de scikit learn
mean = X_train.mean()
std  = X_train.std()
X_train = (X_train - mean) / std
# Normalisation des données de test
X_test  = (X_test  - mean) / std

X_train.head()
#%% md
# ### 3. Construction (définition) du modèle 
# On construite un modèle de régression linéaire. La fonction mise en œuvre par un neurone sans activation est la même que la régression linéaire du chapire 2.:
# $$ f_{\mathbf{w},b}(x^{(i)}) = \mathbf{w}\cdot x^{(i)} + b \tag{1}$$
# 
# La définition d'un modèle exige le choix d'un certains nombre d'options (fonctions):  
# - [Optimizer:](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers ) C'est l'algorithme d'apprentissage utilisé pour optimiser les paramètres
# - [Activation:](https://www.tensorflow.org/api_docs/python/tf/keras/activations): `sigmoid`, `linear`, `relu`, ...
# - [Loss :](https://www.tensorflow.org/api_docs/python/tf/keras/losses) fonctions d'erreurs, `mse`, `BinaryCrossentropy`, `CategoricalCrossentropy`, ...
# - [Metrics :](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) `Accuracy`, `F1Score`,`mae`, `mse`...
#     
# 
#%% md
# ## 3.1 Un modèle plus complexe  
# Construire un réseau comportant : 
# - une couche d'entrée
# - une couche cachée à 32 neurones
# - une couche de sortie 
#%%
# Le modèle 
m = X_train.shape[1] # nombre de variables
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(m, name='input_layer'))
model.add(tf.keras.layers.Dense(units=32, activation='linear'))
model.add(tf.keras.layers.Dense(units=1, activation='linear', name='output'))




#%%
#Compiler le modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae', r2_score])




#%%
#affichage du modèle
model.summary()

#%% md
# ### 4. Entrainement du modèle (Model training)
#%%
# Entrainement du modele
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
print (model.metrics_names)
#%% md
# ### 5. Evaluation des performances du modèle 
# - Afficher tous les scores.
# 
#%%

#%% md
# ### Historique du training (training history)
# On peut afficher l'historique de l'apprentissge. 
# On peut aussi afficher les courbes 
#%%
print(history.params)
print(history.history.keys())
print(history.history['loss'])
#%%
import pandas as pd 
df=pd.DataFrame(data=history.history)
display(df)
#%%
import matplotlib.pyplot as plt

# Plotting the loss curve





# Plotting the r2_score curve

#%% md
# ### 6. Faire des prédictions
# 
#%%
#Small Sata set
my_data = [ 0.126918, 0.417687, 1.374513, -0.502325 ]
real_price = 350.00 # ce que vous devriez obtenir doit être proche de ce prix

#%%
# écrire le code ici 


#%% md
# <details>
# <summary>
#     <font size='3', color='darkgreen'><b>Attention au format de l'entrée, sinon essayez ...</b></font>
# </summary>
#     my_data=np.array(my_data).reshape(1,4)
# 
#%% md
# ### Exercice
# I- Comparer les performances obtenues par les deux modèles 
# - 1) Le perceptron et le modèle multicouches sur les deux datasets?
# - 2) Modifier la structure du réseau dense
#     - augmenter le nombre de neurones par couche
#     - augmenter le nombre de couches. 
#     
# Quel est le réseau qui donne le meilleur résultat ?
#%% md
# II- Refaire les questions de l'exercice en considérant le Dataset ci dessous un peu plus large.
# Le nom des variables est différent du fichier précédent (mais on reste dans la prédiction du prix d'un bien immobilier, ici median_house_value)
#%% md
# #### Large Dataset
#%%
# lecture du fichier texte.
data = pd.read_csv('../data/housing/housing.csv', header=0)

data.dropna(axis=0, inplace=True)
print('Missing Data : ',data.isna().sum().sum(), '  Shape is : ', data.shape)

X = data.drop('median_house_value',  axis=1)
y = data['median_house_value']
#%%
## affichage des noms de colonnes (variables du modèle) 
X.columns