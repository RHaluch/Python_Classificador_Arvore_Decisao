from sklearn import tree, datasets, metrics
import numpy as np

iris = datasets.load_digits()

X = iris.data
Y = iris.target

np.random.seed(0)
n_samples = len(X)
percent = 0.75

order = np.random.permutation(n_samples)
X = X[order]
Y = Y[order]

Y_teste = Y[int(percent*n_samples):]
X_teste = X[int(percent*n_samples):]

Y_treino = Y[:int(percent*n_samples)]
X_treino = X[:int(percent*n_samples)]

clf = tree.DecisionTreeClassifier()

clf.fit(X_treino,Y_treino)

predicao = clf.predict(X_teste)

print(clf.score(X_teste,Y_teste))
matriz=metrics.confusion_matrix(Y_teste, predicao)

for item in matriz:
    print(item)

atributos=['comprimento sepala','largura sepala','comprimento petala','largura petala']
classes=['setosa','versicolor','virginica']
c2=['0','1','2','3','4','5','6','7','8','9']

tree.export_graphviz(clf,"arvore3.dot",class_names=c2)

