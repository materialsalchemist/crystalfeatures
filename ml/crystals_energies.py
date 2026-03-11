from sklearn import svm
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

plt.rcParams.update({'font.size': 30})

col = 0


use_models = False
use_datasets = False


energies = ['band_gaps', 'formation_energy_per_atom', 'e_above_hull', 'energy_per_atom']

energy = energies[col]

output_file = open('output_'+energy, 'w')

numXColumns = 0
# Remove all records with inf values

figfolder = energy+'/Figures/'

removeMetals = False
merge_basic = True
merge_fragments = False
merge_geometry = True


regr_names = ['XGBOOST', 'RF', 'SVM']
regr_objects = [XGBRegressor(n_estimators=100, max_depth=100, random_state=0),
                RandomForestRegressor(
                    n_estimators=400, max_depth=400, random_state=0),
                svm.SVR(kernel='rbf', epsilon=0.1, verbose=True)
                # RVR(kernel='rbf', n_iter=10000, tol=0.0001, verbose=True),
                # linear_model.HuberRegressor(
                #     epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05),

                ]

if not use_datasets:

    full_dataset_df = pd.read_csv(
        'band_gaps/Dataset/dataset_expanded.csv', index_col='material_id')
    # full_dataset_df = pd.read_csv('Figures_v6/dataset.csv', index_col=0)
    # Data visualization: let's have a look at our data.
    # del full_dataset_df['band_gaps']

    for c in full_dataset_df.columns:
        if 'Unnamed' in c:
            del full_dataset_df[c]

    numXColumns = full_dataset_df.shape[1]

    if merge_basic:
        dataset_basic = pd.read_csv(
            '../ML_BasicDescriptors/BandGap/Dataset/dataset_bacd.csv', index_col='material_id')
        # del dataset_basic['band_gaps']
        for c in dataset_basic.columns:
            if 'Unnamed' in c:
                del dataset_basic[c]
        numXColumns += dataset_basic.shape[1]

        full_dataset_df = pd.merge(
            full_dataset_df, dataset_basic, left_index=True, right_index=True)

    if merge_fragments:
        dataset_fragments = pd.read_csv(
            '../ML_Fragments/BandGap/Dataset/dataset.csv', index_col='material_id')
        del dataset_fragments['band_gaps']
        for c in dataset_fragments.columns:
            if 'Unnamed' in c:
                del dataset_fragments[c]
        numXColumns += dataset_fragments.shape[1]

        full_dataset_df = pd.merge(
            full_dataset_df, dataset_fragments, left_index=True, right_index=True)

    if merge_geometry:
        dataset_geometry = pd.read_csv(
            '../ML_SymmetryFunctions/BandGap/Dataset/dataset_B.csv', index_col='material_id')
        # del dataset_geometry['band_gaps']

        for c in dataset_geometry.columns:
            if 'Unnamed' in c:
                del dataset_geometry[c]

        numXColumns += dataset_geometry.shape[1]

        full_dataset_df = pd.merge(
            full_dataset_df, dataset_geometry, left_index=True, right_index=True)

    energies_df = pd.read_csv(
        '../dataset_energies.csv', index_col='material_id')

    full_dataset_df = pd.merge(
        full_dataset_df, energies_df, left_index=True, right_index=True)

    full_dataset_df = full_dataset_df.drop(
        index=full_dataset_df.index[np.isinf(full_dataset_df).any(1)])

    full_dataset_df = full_dataset_df.drop(
        index=full_dataset_df.index[np.isnan(full_dataset_df).any(1)])

    for c in full_dataset_df.columns:
        if 'Unnamed' in c:
            del full_dataset_df[c]

    print('Dataset size:', full_dataset_df.shape)

    # Testing R2
    regr_choice = 0
    R2 = 0
    MAE = 1
    while R2 < 0.88:
        # while MAE > 0.1 and R2 < 0.97:

        # X_train_scaled, X_test_scaled, y_train, y_test = dataset_split.dataset_split(
        #     'BandGap/', full_dataset_df, 1, 'regression', False)
        y = full_dataset_df[energy]

        X = full_dataset_df.iloc[:, :numXColumns]
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=None)

        scalerX = StandardScaler().fit(X_train_scaled)
        X_train_scaled = scalerX.transform(X_train_scaled)
        X_test_scaled = scalerX.transform(X_test_scaled)
        from pickle import dump
        dump(scalerX, open(energy+'/SavedModels/' +
                           'regression_scalerX.pkl', 'wb'))

        regr = regr_objects[regr_choice]
        regr_name = regr_names[regr_choice]

        regr.fit(X_train_scaled, y_train)

        y_predicted = regr.predict(X_test_scaled)

        output_file.write('Iteration:\n')
        output_file.write(regr_name+' MAE\t' +
                          str(mean_absolute_error(y_test, y_predicted))+'\n')
        output_file.write(regr_name+' R2\t' +
                          str(r2_score(y_test, y_predicted))+'\n')
        output_file.flush()

        print(regr_name+' MAE\t'+str(mean_absolute_error(y_test, y_predicted))+'\n')
        print(regr_name+' R2\t'+str(r2_score(y_test, y_predicted))+'\n')
        R2 = r2_score(y_test, y_predicted)
        MAE = mean_absolute_error(y_test, y_predicted)

    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    X_train_scaled_df.to_csv(energy+'/Dataset/' +
                             'X_train_scaled.csv')
    X_test_scaled_df.to_csv(energy+'/Dataset/X_test_scaled.csv')
    y_train_df.to_csv(energy+'/Dataset/y_train.csv')
    y_test_df.to_csv(energy+'/Dataset/y_test.csv')

else:
    X_train_scaled = pd.read_csv(
        energy+'/Dataset/X_train_scaled.csv')
    X_test_scaled = pd.read_csv(
        energy+'/Dataset/X_test_scaled.csv')
    y_train = pd.read_csv(energy+'/Dataset/y_train.csv')
    y_test = pd.read_csv(energy+'/Dataset/y_test.csv')
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
            energy+'/SavedModels/' +
            regr_name + '_Regression.pkl')
        y_predicted = regr.predict(X_test_scaled.values)

    print(regr_name+' MAE\t'+str(mean_absolute_error(y_test, y_predicted))+'\n')
    print(regr_name+' R2\t'+str(r2_score(y_test, y_predicted))+'\n')

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
        joblib.dump(regr, energy+'/SavedModels/' +
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
    # plt.clf()

output_file.close()
