import glmnet
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics

X, y = sklearn.datasets.load_iris(return_X_y=True)
Xtrn, Xtst, ytrn, ytst = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=4)

clf = glmnet.LogitNet()
clf.fit(Xtrn, ytrn)
ypred = clf.predict(Xtst)
acc = sklearn.metrics.accuracy_score(ytst, ypred)

print('glmnet accuracy on iris:', acc)
assert acc > 0.9
