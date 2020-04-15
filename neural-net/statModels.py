import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.metrics
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import tensorflow as tf
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV


def modelfit(alg, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='rmse')

    # Predict training set:
    dtrain_predictions = alg.predict(X_train)

    # Print model report:
    print("\nModel Report")
    final_mse = mean_squared_error(y_train, np.round(dtrain_predictions))
    final_rmse = np.sqrt(final_mse)
    print("Train RMSE : %.4g" % final_rmse)

    dtest_predictions = alg.predict(X_test)
    final_mse = mean_squared_error(y_test, np.round(dtest_predictions))
    final_rmse = np.sqrt(final_mse)
    print("XGBoost RMSE : %.4g" % final_rmse)

    pickle.dump(alg, open('xgblin.pkl', 'wb'))



def main():
    teamAbbr = ["CHN", "PIT", "PHI", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]

    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')

    ## LINEAR REGRESSION ##
    print('Re-training linear regression')
    data = pd.read_csv('combinedData.csv')
    data['Win'] = data['Win'].astype(np.int64)

    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models')
    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'pitchersUsed', 'wonPrev', 'OPS'], axis=1,inplace=True)

    #getCorrelationMatrix(data)

    data.drop(['Win'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                        data['Score'], test_size=0.20,
                                                        random_state=101)
    scoreModel = LinearRegression()
    scoreModel.fit(X_train,y_train)

    predictions = scoreModel.predict(X_test)

    final_mse = mean_squared_error(y_test, np.round(predictions))

    final_rmse = np.sqrt(final_mse)
    print("Linear Regression RMSE: " + str(final_rmse))

    pickle.dump(scoreModel, open('linmodel.pkl', 'wb'))
    print('Re-trained linear regression')
    ## ------------------------------------------------- ##

    ## RANDOM FOREST REGRESSION ##
    print('Re-training random forest regression')
    from sklearn.ensemble import RandomForestRegressor
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    data = pd.read_csv('combinedData.csv')
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models')

    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'pitchersUsed', 'wonPrev', 'OPS'], axis=1, inplace=True)


    data.drop(['Win'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                        data['Score'], test_size=0.20,
                                                        random_state=101)

    #rf = RandomForestRegressor(random_state=42)

    #param_grid = {
    #    'n_estimators': [100, 250, 500],
    #    'max_features': ['auto', 'sqrt', 'log2'],
    #    'max_depth': range(5, 35, 5),
    #    'min_samples_split': [2, 5, 10, 15],
    #    'min_samples_leaf': [1, 2, 4, 6, 8]
    #}

    #tuneParameters(rf, param_grid, X_train, y_train)

    
    rf = RandomForestRegressor(n_estimators=250, bootstrap=True, max_features='auto', max_depth=15, min_samples_split=2, min_samples_leaf=1, random_state=42)

    rf.fit(X_train, y_train)

    train_preds = rf.predict(X_train)

    train_mse = mean_squared_error(y_train, np.round(train_preds))
    train_rmse = np.sqrt(train_mse)

    predictions = rf.predict(X_test)

    final_mse = mean_squared_error(y_test, np.floor(predictions))
    final_rmse = np.sqrt(final_mse)

    print("Random Forest Train RMSE: " + str(train_rmse))
    print("Random Forest Test RMSE: " + str(final_rmse))

    pickle.dump(rf, open('rflin.pkl', 'wb'))
    print('Re-trained random forest regression')
    ## --------------------------------------------------- ##

    ## XGBoost Regression
    print('Re-training XGBoost regression')
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    data = pd.read_csv('combinedData.csv')
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models')
    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'pitchersUsed', 'wonPrev', 'OPS'], axis=1, inplace=True)

    data.drop(['Win'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                        data['Score'], test_size=0.20,
                                                        random_state=101)


    xgb1 = XGBRegressor(learning_rate=.1, n_estimators=250, max_depth=7, min_child_weight=8,
                       gamma=0, reg_lambda=.1, subsample=.8, colsample_bytree=.8, objective='reg:squarederror',
                       nthread=4, seed=27)

    modelfit(xgb1, X_train, y_train, X_test, y_test)

    # param_test1 = {
    #    'max_depth': range(3, 12, 2),
    #    'min_child_weight': range(3, 12, 2),
    #    'subsample': [i / 10.0 for i in range(1, 10,2)],
    #    'colsample_bytree': [i / 10.0 for i in range(1, 10,2)],
    #    'reg_lambda': [0, .001, .01, 0.1, .5, 1]
    # }
    # xgb = XGBRegressor(learning_rate=0.1, n_estimators=250, max_depth=8,
    #                                                min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #                                                objective='reg:squarederror', nthread=4, scale_pos_weight=1,
    #                                                reg_lambda=.5, seed=27, tree_method='gpu_hist', predictor='gpu_predictor')

    #tuneParameters(xgb, param_test1, X_train, y_train)

    ## ----------------------------------------- ##
    print('Re-trained XGBoost regression')
    ## ---------- LOGISTIC REGRESSION ---------- ##
    print('Re-training logistic regression')
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    data = pd.read_csv('combinedData.csv')
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models')

    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data.drop('Win', axis=1),
                                                        data['Win'], test_size=0.20,
                                                        random_state=93)

    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)


    logistic_reg_score_train = logmodel.score(X_train, y_train)
    print("Percentage correct on training set = ", 100. * logistic_reg_score_train, "%")

    predictions = logmodel.predict(X_test)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions))
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)

    # getConfusionMatrix(conf_matrix)
    # getROCCurve(logmodel, X_test, y_test)
    # getFeatureImportances(X_train, y_train)
    print('Re-trained logistic regression')
    pickle.dump(logmodel, open('logmodel.pkl', 'wb'))

    ## ------------------------------------------------------------- ##


def getCorrelationMatrix(data):

    plt.subplots(figsize=(35,35))
    cor = data.corr()
    sns.heatmap(cor, cmap=plt.cm.Reds)
    plt.gcf().subplots_adjust(top=.95,bottom=0.35)
    plt.xticks(rotation=45,ha='right')
    fig = plt.gcf()
    plt.show()
    fig.savefig("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\Correlation_Heatmap.png")
    plt.close('all')

    return

def tuneParameters(model, param_grid, X_train, y_train):
    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=8, verbose=2)
    CV_rfc.fit(X_train, y_train)

    print(CV_rfc.best_params_)

    return

def getConfusionMatrix(matrix):

    ax = plt.subplot()
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu", fmt='d')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Loss', 'Win'])
    ax.yaxis.set_ticklabels(['Loss', 'Win'])
    saveFig = plt.gcf()
    plt.show()

    saveFig.savefig("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\Sample_Conf_Matrix.png")
    plt.close()

    return

def getROCCurve(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    from plot_metric.functions import BinaryClassification

    bc = BinaryClassification(y_test, y_pred, labels=["Win", "Loss"])

    plt.figure(figsize=(5, 5))
    bc.plot_roc_curve()
    plt.show()

    return

def getFeatureImportances(X_train, y_train):
    forest = ExtraTreesClassifier(n_estimators=500)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature Ranking:")

    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

    return

if __name__ == "__main__":
    main()