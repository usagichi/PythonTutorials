from Day3project.wrangling import assemble
from Day3project.regression import simple_linear_regression, enet_model_selection, enet_cv_model_selection
from Day3project.classification import *

X, y = assemble("boston")
model = simple_linear_regression(X, y, plot_figures = True)
model = enet_model_selection(X, y)
model = enet_cv_model_selection(X, y)

X, y = assemble("breast_cancer")
model = train_and_eval_knn(X, y)
model = batch_train_models(X, y)

pass