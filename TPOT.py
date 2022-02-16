#coding: utf-8

'''
TPOT
'''


import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split


# input data
data_input=pd.read_csv('3DCN-2022.csv',sep=',')

labels=data_input['Cap']#[:,np.newaxis]
features=data_input.drop('Cap', axis=1)

X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=0)

tpot = TPOTRegressor(random_state=0,verbosity=2,template='Regressor')

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('TPOT_pipeline_2022.py')
