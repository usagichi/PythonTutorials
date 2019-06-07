from __future__ import division

import os
import pickle
from functools import reduce

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def confusion_matrix(actual, predicted):
    labels = sorted(list(set(actual).intersection(set(predicted))))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(actual)):
        cm[labels.index(actual[i]), labels.index(predicted[i])] += 1
    return cm


def classification_metrics(cm):
    accuracy = np.trace(cm) / np.sum(cm)
    sensitivity = cm[1, 1] / sum(cm[1, :])
    specificity = cm[0, 0] / sum(cm[0, :])
    precision = cm[1, 1] / sum(cm[:, 1])
    return accuracy, sensitivity, specificity, precision


def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues, show=True, title=None, xlabel=None, ylabel=None, rotation=0):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=rotation)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel('Predicted label')
    if ylabel is not None:
        plt.ylabel('True label')
    plt.tight_layout()
    if show:
        plt.show()


def train_and_eval_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    cm = confusion_matrix(y, model.predict(X))
    plot_confusion_matrix(cm, classes=['malignant', 'benign'], title="Confusion matrix",
                          xlabel="Predicted label", ylabel="True label", rotation=45)
    accuracy, sensitivity, specificity, precision = classification_metrics(cm)
    print("Simple KNN metrics: accuracy=%.2f, sens=%.2f, spec=%.2f, prec=%.2f\n" % (
        accuracy, sensitivity, specificity, precision))
    return model


def cv_model_selection(X, y, clazz, **kwargs):
    kf = KFold(n_splits=5)
    best_model, best_accuracy = None, None

    hyperparameter_names = sorted(kwargs.keys())
    hyperparameter_values = [kwargs[name] for name in hyperparameter_names]
    hyperparameter_combinations = list(itertools.product(*hyperparameter_values))

    for combination in hyperparameter_combinations:
        parameter_mapping = {name: combination[idx] for idx, name in enumerate(hyperparameter_names)}

        cv_accuracy = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = [y[idx] for idx in train_index], [y[idx] for idx in test_index]
            model = clazz(**parameter_mapping)
            model.fit(X_train, y_train)
            cv_accuracy.append(accuracy_score(y_test, model.predict(X_test)))

        current_accuracy = reduce(lambda x, y: x + y, cv_accuracy) / len(cv_accuracy)
        if best_accuracy is None or best_accuracy < current_accuracy:
            best_accuracy = current_accuracy
            best_model = model

    final_model = clazz(**best_model.get_params())
    final_model.fit(X, y)
    return final_model


def batch_train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    knn = cv_model_selection(X_train, y_train, KNeighborsClassifier, n_neighbors=[5, 10, 15, 20])
    svc = cv_model_selection(X_train, y_train, SVC, kernel=["linear", "rbf"], C=[1.0, 5.0])
    rfc = cv_model_selection(X_train, y_train, RandomForestClassifier, n_estimators=[10, 100, 1000])

    os.makedirs(os.path.join(os.curdir, 'dumps'))
    with open("dumps/knn", "wb") as f:
        pickle.dump(knn, f)
    with open("dumps/svc", "wb") as f:
        pickle.dump(svc, f)
    with open("dumps/rfc", "wb") as f:
        pickle.dump(rfc, f)

    with open("dumps/knn", "rbU") as f:
        knn = pickle.load(f)
    with open("dumps/svc", "rbU") as f:
        svc = pickle.load(f)
    with open("dumps/rfc", "rbU") as f:
        rfc = pickle.load(f)

    batch_eval_models(X_test, y_test, knn, svc, rfc)
    return None


def batch_eval_models(X, y, knn, svc, rfc):
    knn_cm = confusion_matrix(y, knn.predict(X))
    svc_cm = confusion_matrix(y, svc.predict(X))
    rfc_cm = confusion_matrix(y, rfc.predict(X))
    plot_confusion_matrix_comparison(knn_cm, svc_cm, rfc_cm)

    score_labels = ["Accuracy", "Sensitivity", "Specificity", "Precision"]
    knn_scores = list(classification_metrics(knn_cm))
    svc_scores = list(classification_metrics(svc_cm))
    rfc_scores = list(classification_metrics(rfc_cm))
    plot_classification_metrics_bar_chart(score_labels, knn_scores, svc_scores, rfc_scores)


def plot_confusion_matrix_comparison(knn_cm, svc_cm, rfc_cm):
    cms = [knn_cm, svc_cm, rfc_cm]
    titles = ["KNN", "SVC", "RFC"]
    classes = ['malignant', 'benign']

    cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]
    figure, subplots = plt.subplots(1, 3)
    figure.set_size_inches(18, 6)

    for i in range(3):
        plt.sca(subplots[i])
        plot_confusion_matrix(cm=cms[i], classes=classes, title=titles[i], cmap=cmaps[i], show=False)

    subplots[0].set_ylabel("True label")
    for i in range(3):
        subplots[i].set_xlabel("Predicted label")

    plt.suptitle("Confusion matrices for the various models")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()


def plot_classification_metrics_bar_chart(labels, knn_scores, svc_scores, rfc_scores):
    ind = np.arange(4)
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, knn_scores, width, label="KNN", color='r')
    rects2 = ax.bar(ind + width, svc_scores, width, label="SVC", color='g')
    rects3 = ax.bar(ind + 2 * width, rfc_scores, width, label="RFC", color='b')

    ax.set_ylabel('Scores')
    ax.set_title('Classifier scores')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, '%.2f' % height, ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.show()
