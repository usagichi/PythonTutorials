from functools import reduce

import matplotlib.pyplot as plt
from itertools import product
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def print_coef(coef, feature_names):
    p = len(feature_names)
    print("Feature |      Coef")
    print("-------------------")
    for idx in range(p):
        print("%7s\t:\t%7.3f" % (feature_names[idx], coef[idx]))


def plot_coef_bars(coef, feature_names):
    fig, ax = plt.subplots()
    ax.bar(range(len(coef)), [abs(x) for x in coef])
    ax.set_xticks(range(len(coef)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_xlabel("Features")
    ax.set_ylabel("Coefficient magnitude")
    plt.title("Magnitude of predictor coefficients")
    plt.tight_layout()
    plt.show()


def prediction_scatter_plot(actual, predicted):
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted)
    ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.tight_layout()
    plt.show()


def target_by_predictor_scatter_plot(target, predictor_name, predictor_values):
    fig, ax = plt.subplots()
    ax.scatter(predictor_values, target)
    ax.set_xlabel(predictor_name)
    ax.set_ylabel('Target')
    plt.title("Target values by %s" % predictor_name)
    plt.tight_layout()
    plt.show()


def simple_linear_regression(X, y, plot_figures=True):
    model = LinearRegression()
    model.fit(X, y)

    coef = model.coef_.tolist()
    print_coef(coef, X.columns)
    prediction = model.predict(X)

    if plot_figures:
        plot_coef_bars(coef, X.columns)
        prediction_scatter_plot(y, prediction)

        most_relevant_predictor_index = coef.index(max(coef, key=lambda y: abs(y)))
        most_relevant_predictor_name = X.columns[most_relevant_predictor_index]
        most_relevant_predictor_values = X[most_relevant_predictor_name].values
        target_by_predictor_scatter_plot(y, most_relevant_predictor_name, most_relevant_predictor_values)

    print("Linear Regression MSE is %.2f\n" % mean_squared_error(y, prediction))
    return model


alphas = [1e-2, 1e-1, 1, 10, 100, 1000]
l1_ratios = [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95, .99]


def train_elastic_net(X, y, alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, y)
    return model


def enet_model_selection(X, y):
    best_model, best_mse = None, None
    hyperparameter_combinations = list(product(alphas, l1_ratios))
    for alpha, l1_ratio in hyperparameter_combinations:
        model = train_elastic_net(X, y, alpha, l1_ratio)
        mse = mean_squared_error(y, model.predict(X))
        if best_mse is None or best_mse > mse:
            best_mse = mse
            best_model = model

    print_coef(best_model.coef_.tolist(), X.columns)
    print(
        "Best Elastic Net MSE is %.2f, alpha=%.2f; l1_ratio=%.2f\n" % (best_mse, best_model.alpha, best_model.l1_ratio))
    return best_model


def enet_cv_model_selection(X, y):
    kf = KFold(n_splits=5)
    best_model, best_mse = None, None
    hyperparameter_combinations = list(product(alphas, l1_ratios))
    for alpha, l1_ratio in hyperparameter_combinations:

        cv_mse = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = [y[idx] for idx in train_index], [y[idx] for idx in test_index]
            model = train_elastic_net(X_train, y_train, alpha, l1_ratio)
            y_pred = model.predict(X_test)
            cv_mse.append(mean_squared_error(y_test, y_pred))

        cur_mse = reduce(lambda x, y: x + y, cv_mse) / len(cv_mse)
        if best_mse is None or best_mse > cur_mse:
            best_mse = cur_mse
            best_model = model

    print_coef(best_model.coef_.tolist(), X.columns)
    print("Best Elastic Net cross-validated MSE %.2f, alpha=%.2f; l1_ratio=%.2f\n" % (
        best_mse, best_model.alpha, best_model.l1_ratio))
    return best_model
