import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics


def modelisation(X, y, model=LogisticRegression(random_state=42)):
    # applying one hot encoding to Pclass which is categorical variable
    X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'isChild', 'title'], drop_first=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # trainning model on train set
    model.fit(X_train, y_train)
    # make prediction
    y_pred = model.predict(X_test)

    train_score = model.score(X_train, y_train)
    print("Train score: {:0.2%}".format(train_score))
    
    # test score
    test_score = model.score(X_test, y_test)
    print("Test score: {:0.2%}".format(test_score))

    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")