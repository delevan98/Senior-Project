import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import tensorflow as tf


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

    data = pd.read_csv('combinedData.csv')

    #data.drop(['Unnamed: 0'], axis=1, inplace=True)

    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'pitchersUsed'], axis=1, inplace=True)
    #data['wonPrev'].fillna(0, inplace=True)

    data.drop(['Win'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                        data['Score'], test_size=0.20,
                                                        random_state=101)

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    final_mse = mean_squared_error(y_test, np.floor(predictions))
    final_rmse = np.sqrt(final_mse)

    print("Random Forest RMSE: " + str(final_rmse))

    #pickle.dump(rf, open('linmodel.pkl', 'wb'))

    ## --------------------------------------------------- ##

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