from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def confusion_matrix_model(model_used, x_test, y_test):
    cm = confusion_matrix(y_test, model_used.predict(x_test))
    col = ["Predicted UP","Predicted DOWN"]
    cm = pd.DataFrame(cm)
    cm.columns = ["Predicted UP","Predicted DOWN"]
    cm.index = ["Actual UP","Actual DOWN"]
    cm[col] = np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)

    return cm


def auc_score(x_test, y_test, model, has_proba=True):
    if has_proba:
        fpr, tpr, thresh = skplt.metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
    else:
        fpr, tpr, thresh = skplt.metrics.roc_curve(y_test, model.decision_function(x_test))
    x = fpr
    y = tpr
    auc = skplt.metrics.auc(x, y)
    return auc


def plt_roc_curve(x_test, y_test, name, model, has_proba=True):
    if has_proba:
        fpr, tpr, thresh = skplt.metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
    else:
        fpr, tpr, thresh = skplt.metrics.roc_curve(y_test, model.decision_function(x_test))
    x = fpr
    y = tpr
    auc = skplt.metrics.auc(x, y)
    plt.plot(x,y,label='ROC curve for %s (AUC = %0.2f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def importance_of_features_xgb(model, feature_ls):
    features = pd.DataFrame()
    features['feature'] = feature_ls
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    plt.figure(figsize=(20, 6))
    plt.bar(features.index, features['importance'])
    plt.show()