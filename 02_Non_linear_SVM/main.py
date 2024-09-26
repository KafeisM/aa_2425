import math

import sklearn
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# Ja no necessitem canviar les etiquetes, Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

#Implementacions kernels propis
def kernel_lineal(x1, x2):
    return x1.dot(x2.T)

gamma = 1.0/(X_transformed.shape[1] * X_transformed.var())
def kernel_rbf(x1, x2):
    return np.exp(-gamma * distance_matrix(x1,x2)**2)

def kernel_poly(x1,x2,degree=3):
    return (gamma*x1.dot(x2.T))**degree

#KERNEL LINEAL
svm = SVC(C=1.0, kernel='linear',random_state=33)
svm.fit(X_transformed, y_train, sample_weight=None)
y_pred_svm = svm.predict(X_test_transformed)

Meu_svm = SVC(C=1.0, kernel=kernel_lineal,random_state=33)
Meu_svm.fit(X_transformed, y_train, sample_weight=None)
y_pred_Meusvm =Meu_svm.predict(X_test_transformed)

# Imprimir resultats
print("Resultats per SVM kernel lineal de sklearn:")
svm_prec = sklearn.metrics.precision_score(y_test, y_pred_svm)
print(f"Precisión: {svm_prec:.4f}")

print("\nResultats per SVM amb kernel lineal personalitzat:")
meu_svm_prec = sklearn.metrics.precision_score(y_test, y_pred_Meusvm)
print(f"Precisión: {meu_svm_prec:.4f}")

#KERNEL GAUSSIÀ
svm = SVC(C=1.0, kernel='rbf',random_state=33)
svm.fit(X_transformed, y_train, sample_weight=None)
y_pred_svm = svm.predict(X_test_transformed)

Meu_svm = SVC(C=1.0, kernel=kernel_rbf,random_state=33)
Meu_svm.fit(X_transformed, y_train, sample_weight=None)
y_pred_Meusvm =Meu_svm.predict(X_test_transformed)

# Imprimir resultats
print("\nResultats per SVM kernel gaussia de sklearn:")
svm_prec = sklearn.metrics.precision_score(y_test, y_pred_svm)
print(f"Precisión: {svm_prec:.4f}")

print("\nResultats per SVM amb kernel gaussia personalitzat:")
meu_svm_prec = sklearn.metrics.precision_score(y_test, y_pred_Meusvm)
print(f"Precisión: {meu_svm_prec:.4f}")

#KERNEL POLINOMIAL
svm = SVC(C=1.0, kernel='poly',random_state=33)
svm.fit(X_transformed, y_train, sample_weight=None)
y_pred_svm = svm.predict(X_test_transformed)

Meu_svm = SVC(C=1.0, kernel=kernel_poly,random_state=33)
Meu_svm.fit(X_transformed, y_train, sample_weight=None)
y_pred_Meusvm =Meu_svm.predict(X_test_transformed)

# Imprimir resultats
print("\nResultats per SVM kernel polinòmic de sklearn:")
svm_prec = sklearn.metrics.precision_score(y_test, y_pred_svm)
print(f"Precisión: {svm_prec:.4f}")

print("\nResultats per SVM amb kernel polinòic personalitzat:")
meu_svm_prec = sklearn.metrics.precision_score(y_test, y_pred_Meusvm)
print(f"Precisión: {meu_svm_prec:.4f}")

#Dibuixar froteres de decisó





