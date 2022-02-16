
# coding: utf-8


import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score,KFold
from sklearn import preprocessing, svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


tpot_data = pd.read_csv('3DCN-2022.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Cap', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Cap'], random_state=0)

exported_pipeline = GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=10, max_features=0.25, min_samples_leaf=1, min_samples_split=9, n_estimators=100, subsample=1.0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
results_train = exported_pipeline.predict(training_features)

shap.initjs()
explainer = shap.TreeExplainer(exported_pipeline)

y_base = explainer.expected_value
print(y_base)

predictt = exported_pipeline.predict(training_features)
print(predictt.mean())

shap_values = explainer.shap_values(features)
np.savetxt("....csv", shap_values, delimiter=",")
shap.plots.scatter(shap_values[:,4])
fig=shap.summary_plot(shap_values, features,show=False) 
fig2=shap.summary_plot(shap_values, features, plot_type="bar") 
plt.show()
