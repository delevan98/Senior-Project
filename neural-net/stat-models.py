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
    alg.fit(X_train, y_train, eval_metric='auc')

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
    print("Test RMSE : %.4g" % final_rmse)



def main():
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    teamAbbr = ["CHN", "PIT", "PHI", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]

    frames = []
    for x in range(30):
        frames.append(pd.read_csv(teamAbbr[x] + '_Full.csv'))

    result = pd.concat(frames)
    result['wonPrev'].fillna(0, inplace=True)

    ## LINEAR REGRESSION ##
    result.to_csv('combinedData.csv', index=False)

    data = pd.read_csv('combinedData.csv')
    data['Win'] = data['Win'].astype(np.int64)

    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER'], axis=1,inplace=True)
    # Using Pearson Correlation
    plt.subplots(figsize=(35,35))
    cor = data.corr()
    sns.heatmap(cor, cmap=plt.cm.Reds)
    plt.gcf().subplots_adjust(top=.95,bottom=0.35)
    plt.xticks(rotation=45,ha='right')
    fig = plt.gcf()
    plt.show()
    fig.savefig("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\Correlation_Heatmap.png")
    plt.close('all')

    data.drop(['Win'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                        data['Score'], test_size=0.20,
                                                        random_state=101)
    scoreModel = LinearRegression()
    scoreModel.fit(X_train,y_train)

    predictions = scoreModel.predict(X_test)
    #print("Predictions: ", np.floor(predictions))

    final_mse = mean_squared_error(y_test, np.floor(predictions))
    print(final_mse)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)

    pickle.dump(scoreModel, open('linmodel.pkl', 'wb'))

    ## ------------------------------------------------- ##

    ## RANDOM FOREST REGRESSION ##

    from sklearn.ensemble import RandomForestRegressor

    #data = pd.read_csv('combinedData.csv')

    #data.drop(['Unnamed: 0'], axis=1, inplace=True)

    #data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'pitchersUsed'], axis=1, inplace=True)
    #data['wonPrev'].fillna(0, inplace=True)

    #data.drop(['Win'], axis=1, inplace=True)
    #X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
     #                                                   data['Score'], test_size=0.20,
     #                                                   random_state=101)

    #rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    #rf.fit(X_train, y_train)

    #predictions = rf.predict(X_test)

    #final_mse = mean_squared_error(y_test, np.floor(predictions))
    #final_rmse = np.sqrt(final_mse)

    #print("Random Forest RMSE: " + str(final_rmse))

    #pickle.dump(rf, open('linmodel.pkl', 'wb'))

    ## --------------------------------------------------- ##

    ## XGBoost Regression

    data = pd.read_csv('combinedData.csv')
    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'pitchersUsed'], axis=1, inplace=True)

    data.drop(['Win'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                        data['Score'], test_size=0.20,
                                                        random_state=101)


    xgb1 = XGBRegressor(learning_rate=.1, n_estimators=1000, max_depth=5, min_child_weight=3,
                       gamma=0, reg_lambda=.5, subsample=.8, colsample_bytree=.8, objective='reg:squarederror',
                       scale_pos_weight=1, seed=27)

    modelfit(xgb1, X_train, y_train, X_test, y_test)

    #param_test1 = {
        #'max_depth': [6,7,8],
        #'min_child_weight': [2,3,4],
        #'gamma': [i / 10.0 for i in range(0, 5)],
        #'subsample': [i / 100.0 for i in range(65, 80, 5)],
        #'colsample_bytree': [i / 100.0 for i in range(95, 105,5 )],
        #'reg_alpha': [0, .001, 1e-5, 1e-2, .05, 0.1, 1, 100]
    #}

    #gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
    #                                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #                                                objective='reg:squarederror', nthread=4, scale_pos_weight=1,
    #                                                seed=27),
    #                        param_grid=param_test1, scoring='neg_root_mean_squared_error', n_jobs=4, cv=5)

    #gsearch1.fit(X_train, y_train)
    #index = np.where(gsearch1.cv_results_['rank_test_score'] == np.amin(gsearch1.cv_results_['rank_test_score']))
    #print(gsearch1.cv_results_['params'][int(index[0])])

    ## ----------------------------------------- ##

    ## LOGISTIC REGRESSION ##

    data = pd.read_csv('combinedData.csv')
    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER'], axis=1, inplace=True)
    #data['wonPrev'].fillna(0, inplace=True)
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

    #plt.figure(figsize=(10,10))
    ax = plt.subplot()
    sns.heatmap(pd.DataFrame(conf_matrix),annot=True, cmap="YlGnBu", fmt='d')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Loss', 'Win'])
    ax.yaxis.set_ticklabels(['Loss', 'Win'])
    saveFig = plt.gcf()
    plt.show()

    saveFig.savefig("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\Sample_Conf_Matrix.png")
    plt.close()

    from sklearn import metrics

    y_pred = logmodel.predict_proba(X_test)[:,1]
    from plot_metric.functions import BinaryClassification
    # Visualisation with plot_metric
    bc = BinaryClassification(y_test, y_pred, labels=["Win", "Loss"])

    plt.figure(figsize=(5, 5))
    bc.plot_roc_curve()
    plt.show()

    forest = ExtraTreesClassifier(n_estimators=500)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature Ranking:")

    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

    if(saveModel(logmodel) == 1):
        print("Successfully saved model!")

    else:
        print("Model failed to save!")

    ## ------------------------------------------------------------- ##

def saveModel(model):
    pickle.dump(model, open('logmodel.pkl', 'wb'))
    return 1


if __name__ == "__main__":
    main()