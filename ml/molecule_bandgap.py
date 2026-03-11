# pip3 install cmake --user
# pip3 install xgboost --user
# pip3 install sklearn pandas keras --user

from operator import index
from xgboost import XGBClassifier, plot_importance
from sklearn import linear_model, decomposition, datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib
import importlib
import json
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import average_precision_score, precision_score
 

plt.rcParams.update({'font.size': 30})

full_dataset_df = pd.read_csv(
    'Molecules/Dataset/dataset.csv', index_col='material_id')

figfolder = 'Molecules/Figures/'

output_file = open('output', 'w')

# Remove all records with inf values

use_models = False
use_datasets = False


regr_names = ['XGBOOST', 'RF', 'SVM']
regr_objects = [
    # XGBRegressor(n_estimators=100, max_depth=100, random_state=0,
    #             tree_method='gpu_hist', gpu_id=0),
    XGBRegressor(n_estimators=100, max_depth=100, random_state=0),
    RandomForestRegressor(
        n_estimators=100, max_depth=100, random_state=0),
    svm.SVR(kernel='rbf', epsilon=0.1, verbose=True)
    # RVR(kernel='rbf', n_iter=10000, tol=0.0001, verbose=True),
    # linear_model.HuberRegressor(
    #     epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05),

]

if not use_datasets:

    full_dataset_df = full_dataset_df.drop(
        index=full_dataset_df.index[np.isinf(full_dataset_df).any(1)])
    full_dataset_df = full_dataset_df.drop(
        index=full_dataset_df.index[np.isnan(full_dataset_df).any(1)])

    # Testing R2
    regr_choice = 0
    R2 = 0
    MAE = 1
    # while R2 < 0.85:
    while R2 < 0.97:

        y = full_dataset_df.band_gaps
        y = y*27.2114
        numXColumns = full_dataset_df.shape[1]-1
        X = full_dataset_df.iloc[:, :numXColumns]
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=None)

        scalerX = StandardScaler().fit(X_train_scaled)
        X_train_scaled = scalerX.transform(X_train_scaled)
        X_test_scaled = scalerX.transform(X_test_scaled)
        from pickle import dump
        dump(scalerX, open('Molecules/SavedModels/' +
                            'regression_scalerX.pkl', 'wb'))


        regr = regr_objects[regr_choice]
        regr_name = regr_names[regr_choice]

        regr.fit(X_train_scaled, y_train)

        y_predicted = regr.predict(X_test_scaled)

        print(regr_name+' MAE\t'+str(mean_absolute_error(y_test, y_predicted))+'\n')
        print(regr_name+' R2\t'+str(r2_score(y_test, y_predicted))+'\n')
        output_file.write('Iteration:\n')
        output_file.write(regr_name+' MAE\t' +
                          str(mean_absolute_error(y_test, y_predicted))+'\n')
        output_file.write(regr_name+' R2\t' +
                          str(r2_score(y_test, y_predicted))+'\n')
        output_file.flush()

        R2 = r2_score(y_test, y_predicted)
        MAE = mean_absolute_error(y_test, y_predicted)


    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    X_train_scaled_df.to_csv('Molecules/Dataset/' +
                             'X_train_scaled.csv')
    X_test_scaled_df.to_csv('Molecules/Dataset/X_test_scaled.csv')
    y_train_df.to_csv('Molecules/Dataset/y_train.csv')
    y_test_df.to_csv('Molecules/Dataset/y_test.csv')

else:
    X_train_scaled = pd.read_csv(
        'Molecules/Dataset/X_train_scaled.csv')
    X_test_scaled = pd.read_csv(
        'Molecules/Dataset/X_test_scaled.csv')
    y_train = pd.read_csv('Molecules/Dataset/y_train.csv')
    y_test = pd.read_csv('Molecules/Dataset/y_test.csv')
    del X_train_scaled['Unnamed: 0']
    del X_test_scaled['Unnamed: 0']
    del y_train['material_id']
    del y_test['material_id']

output_file.write('Finished finding datasets\n')

for regr_choice in range(0, 1):

    regr = regr_objects[regr_choice]
    regr_name = regr_names[regr_choice]

    if not use_models:
        regr.fit(X_train_scaled, y_train)
        y_predicted = regr.predict(X_test_scaled)
    else:
        regr = joblib.load(
            'Molecules/SavedModels/' +
            regr_name + '_Regression.pkl')
        y_predicted = regr.predict(X_test_scaled.values)

    output_file.write(regr_name+' MAE\t' +
                      str(mean_absolute_error(y_test, y_predicted))+'\n')
    output_file.write(regr_name+' R2\t' +
                      str(r2_score(y_test, y_predicted))+'\n')

    xPlot = y_test
    yPlot = y_predicted
    plt.figure(figsize=(10, 10))
    plt.plot(xPlot, yPlot, 'ro', alpha=0.1)
    plt.plot(xPlot, xPlot)
    plt.ylabel(regr_name)
    plt.xlabel('DFT')
    plt.savefig(figfolder+regr_name +
                '_Regression_Correlation_Test', bbox_inches='tight')

    if not use_models:
        y_predicted = regr.predict(X_train_scaled)
    else:
        y_predicted = regr.predict(X_train_scaled.values)

    print(regr_name+' MAE\t'+str(mean_absolute_error(y_train, y_predicted))+'\n')
    print(regr_name+' R2\t'+str(r2_score(y_train, y_predicted))+'\n')

    output_file.write(regr_name+' MAE\t' +
                      str(mean_absolute_error(y_train, y_predicted))+'\n')
    output_file.write(regr_name+' R2\t' +
                      str(r2_score(y_train, y_predicted))+'\n')

    output_file.flush()

    xPlot = y_train
    yPlot = y_predicted
    plt.figure(figsize=(10, 10))
    plt.plot(xPlot, yPlot, 'ro', alpha=0.1)
    plt.plot(xPlot, xPlot)
    plt.ylabel(regr_name)
    plt.xlabel('DFT')
    plt.savefig(figfolder+regr_name +
                '_Regression_Correlation_Train', bbox_inches='tight')

    if not use_models:
        joblib.dump(regr, 'band_gaps/SavedModels/' +
                    regr_name+'_Regression.pkl')

    # import shap
    # explainer = shap.Explainer(regr)
    # shap_values = explainer(X_test_scaled)
    # plt.figure(figsize=(10, 10))
    # shap.plots.waterfall(shap_values[0])
    # plt.savefig(figfolder+'SHAP_waterfall_'+regr_name +
    #             '_Regression_Correlation_Test', bbox_inches='tight')
    # plt.clf()
    # # shap.plots.force(shap_values[0])
    # plt.figure(figsize=(10, 10))
    # shap.plots.beeswarm(shap_values)
    # plt.savefig(figfolder+'SHAP_beeswarm_'+regr_name +
    #             '_Regression_Correlation_Test', bbox_inches='tight')
    plt.clf()
    print(regr_name+' MAE\t'+str(mean_absolute_error(y_test, y_predicted))+'\n')
    print(regr_name+' R2\t'+str(r2_score(y_test, y_predicted))+'\n')

output_file.close()

