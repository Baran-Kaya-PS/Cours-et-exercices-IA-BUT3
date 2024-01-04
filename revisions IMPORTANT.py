#%% md
# # TP5.2 -  Exercice (Récap)
# On reprend le fichier titanic. La démarche qu'il faut suivre pour la mise en place d'un projet de machine learning est la suivante : 
# - Analyse des données du Dataset
#   - Analayse de la forme :
#       - Vérifier la taille du Dataset X
#       - Identifier les variables et les labels et leurs types
#       - Visualiser quelques données
#       - Analyser les données manquantes
#       
#  - Pré-traitement des données
#     - Elimination des colonnes inutiles
#     - Traiter es valeurs manquantes
#     - Encoder les colonnes non numériques
#     - Selection de variables utiles 
#     - Afficher (uniquement les courbes à deux dimensions, par exemple la colonne 4 `X[:, 3]`, en fonction de la colonne 3, `X[:, 2]`)
#     - Création des Train set et Test set 
#     - Normaliser des données des différentes colonnes 
# 
# - Définition du modèle de machine learning adéquat à la tâche (données)
# - Evaluation et calcul de performances (ATTENTION Il faut choisir la bonne métrique).
# 
# 
# ## Questions 
# En s'appuyant sur la démarche décrite ci-dessus, comparer les modèles dans les cas suivants :
# - donnéees normalisées versus non normalisées
# - données manquantes traitées versus non traitées
# sur les deux données normalisées et non normalisées et évaluation de leurs performances.
# - comparer différents (2 alogos d'apprentissage). <br>
# 
# PS :
# *** ATTENTION n'oubiez pas de normaliser aussi les données de test. 
# La normalisation des données ne peut se faire que sur les données d'apprentissage. 
# <span style="color:red"> Il ne faut jamais normaliser avant de spliter les données.
# Les données de test ne sont pas connues, un algo. d'apprentissage ne doit jamais utliser une quelconque donnée qui vient des données de test sinon **BIAIS*</span>
# 
# <span style="color:green"> La normalisatin doit donc utiliser le même modèle (```transformer```) que celui utilisé dans la phase d'apprentissage ET SURTOUT la même échelle.</span> 
# 
# Rappel : pour les données d'entrainement on uilise la méthode ```fit_tranform()```et pour les données de test, ```transform()```
# 
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
#%% md
# ### 1 Le dataset
# - Lecture du Dataset et Analyse des données
#%%
def load_titanic() -> pd.DataFrame:
    """Load the titanic dataset.
    Returns:
        DataFrame: the titanic dataset.
    """
    titanic = pd.read_csv('./data/titanic.csv')
    return titanic

raw_titanic_data = load_titanic()
#%% md
# #### Coup d'oeil rapide sur le contenu .head()
#%%
raw_titanic_data.head(3)
#%% md
# #### Vérification taille, type des données 
# 
#%%
raw_titanic_data.info()
#%% md
# #### Types variables
#%% md
# #### Analyse des données manquantes  
# - Combien de valeurs manquantes (selon l'objet renvoyé, pandas ou numpy)
# - On peut caculer le pourcentage de valeurs manquantes  *data.isnull().sum()/data.shape[0])* 
#%%
raw_titanic_data.isna().sum()/raw_titanic_data.shape[0]*100
#%% md
# #### Visualisation (Survived le label en fonction des variables)
#%%
sns.pairplot(raw_titanic_data, hue='Survived')
#%% md
# #### Matrice de correlation 
#%%
sns.heatmap(raw_titanic_data.corr(), annot=True)
#%% md
# ### 2. Pré-traitement des données 
#%% md
# #### Elimination colonnes inutiles
# - Utiliser uniquement 'Survived', 'Pclass', 'Sex', 'Age', 'Fare'
#%%
titanic = raw_titanic_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
X_features = ["Pclass", "Sex", "Age", "Fare"]
titanic["Sex"] = titanic['Sex'].replace(['male', 'female'], [0, 1])
titanic.head(3)


#%% md
# #### Traiter les valeurs manquantes
# - Deux cas : utiliser un imputer ou supprimer toutes les lignes vides
#%%
#Avant quoique ce soit on sépare les données en train et test
from sklearn.model_selection import train_test_split
train, test = train_test_split(titanic, test_size=0.2, random_state=42)

#%%
train_without_nan = train.dropna()
test_without_nan = test.dropna() #dropna supprime les lignes qui ont des valeurs manquantes
#%%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
imputer.fit(train[X_features]) # le imputer sert a remplacer les valeurs manquantes par la moyenne
y_with_all_values = train["Survived"]
y_test_with_all_values = test["Survived"]
train_imputed = imputer.transform(train[X_features])
test_imputed = imputer.transform(test[X_features])


#%% md
# #### Encodage des données non numériques ?
#%%
#Pas besoin l'encodage des données non numériques car on a déjà fait le remplacement des valeurs sexes
# mais pour encoder on utilise la méthode LabelEncoder
#%% md
# #### Normalisation 
# Attention la normalisation à la normalisation des données de test.
#%%
#Normalisation des données
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_imputed)
train_scaled_imputed = scaler.transform(train_imputed)
test_scaled_imputed = scaler.transform(test_imputed)

train_scaled_imputed
#%% md
# ### Pré-processing des données
#%%


#%% md
# #### Visualisation de de queques données (normalisées et non non normalisées) ?
#%%
plt.scatter(range(0, len(train_scaled_imputed[:,2])), train_scaled_imputed[:,2])
plt.scatter(range(0,len(train["Age"])), train["Age"])
plt.title("Age")

#%%
plt.scatter(range(0, len(train_scaled_imputed[:,3])), train_scaled_imputed[:,3])
plt.scatter(range(0,len(train["Fare"])), train["Fare"])
#%%
plt.scatter(range(0, len(train_scaled_imputed[:,0])), train_scaled_imputed[:,0])
plt.scatter(range(0,len(train["Pclass"])), train["Pclass"])
plt.title("Pclass")
#%%
plt.scatter(range(0, len(train_scaled_imputed[:,1])), train_scaled_imputed[:,1])
plt.scatter(range(0,len(train["Sex"])), train["Sex"])

#%% md
# ## Modélisation
#%%
#Modelisation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
n=5
log_reg = KNeighborsClassifier(n_neighbors=n)
#%%
log_reg_naive = KNeighborsClassifier(n_neighbors=n)
#%% md
# ### Entrainement
#%%
log_reg_naive.fit(train_without_nan[X_features],train_without_nan["Survived"]) # no préprocessing (pas de normalisation, pas de traitement des valeurs manquantes)
log_reg.fit(train_scaled_imputed, y_with_all_values) # préprocessing : normalisation et traitement des valeurs manquantes
# pour normaliser -> fit_transform sur les données d'entrainement avec le scaler
# pour les valeurs manquantes -> fit sur les données d'entrainement avec l'imputer
#%% md
# ### Evaluation du modèle
#%%
log_reg_naive.score(test_without_nan[X_features], test_without_nan["Survived"])
#%% md
# 
#%%
log_reg.score(test_scaled_imputed, y_test_with_all_values) 
#%%
# modèle pour la classification
"""
    Logistic Regression: sklearn.linear_model.LogisticRegression
Decision Trees: sklearn.tree.DecisionTreeClassifier
Random Forest: sklearn.ensemble.RandomForestClassifier
Support Vector Machines (SVM): sklearn.svm.SVC
k-Nearest Neighbors (k-NN): sklearn.neighbors.KNeighborsClassifier
Naive Bayes: sklearn.naive_bayes.GaussianNB (for Gaussian Naive Bayes) or sklearn.naive_bayes.MultinomialNB (for Multinomial Naive Bayes)
Gradient Boosting: sklearn.ensemble.GradientBoostingClassifier
AdaBoost: sklearn.ensemble.AdaBoostClassifier
Neural Networks: sklearn.neural_network.MLPClassifier
Linear Discriminant Analysis (LDA): sklearn.discriminant_analysis.LinearDiscriminantAnalysis
Quadratic Discriminant Analysis (QDA): sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
Stochastic Gradient Descent (SGD): sklearn.linear_model.SGDClassifier
Ridge Classifier: sklearn.linear_model.RidgeClassifier
Perceptron: sklearn.linear_model.Perceptron
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
# Make a function that fit and score a list of models on a given dataset
def fit_and_score(models, X_train, y_train, X_test, y_test):
    """Fit and score a list of models on a given dataset
    Args:
        models (list): a list of models
        X_train (DataFrame): the training set
        y_train (Series): the training labels
        X_test (DataFrame): the test set
        y_test (Series): the test labels
    """
    for model in models:
        model.fit(X_train, y_train)
        print(f"{model.__class__.__name__} score: {model.score(X_test, y_test)}")
models = [
    KNeighborsClassifier(n_neighbors=5),  # Adjust n_neighbors as needed
    LogisticRegression(max_iter=1000),  # Adjust max_iter as needed
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),  # Adjust n_estimators and max_depth as needed
    SVC(C=1, kernel='rbf', gamma='scale'),  # Adjust C, kernel, and gamma as needed
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),  # Adjust hidden_layer_sizes and max_iter as needed
    DecisionTreeClassifier(max_depth=5, random_state=42),  # Adjust max_depth as needed
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),  # Adjust n_estimators, learning_rate, and max_depth as needed
    AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),  # Adjust n_estimators and learning_rate as needed
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(max_iter=1000, random_state=42),  # Adjust max_iter as needed
    RidgeClassifier(),
    Perceptron(max_iter=1000, random_state=42),  # Adjust max_iter as needed
    LinearRegression()
]
fit_and_score(models, train_scaled_imputed, y_with_all_values, test_scaled_imputed, y_test_with_all_values)
#%%
# Models pour la régression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor

models = [ #models de regression
    LinearRegression(),
    Lasso(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    KNeighborsRegressor(),
    ElasticNet(),
    HuberRegressor(),
    TheilSenRegressor()
]

#%%
fit_and_score(models, train_scaled_imputed, y_with_all_values, test_scaled_imputed, y_test_with_all_values)