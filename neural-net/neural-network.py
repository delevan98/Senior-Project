import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

import glob
import sys
import csv


def main():
    os.chdir('C:\\Users\\Mike Delevan\\git\\Senior-Project\\data-scraper')

    teamAbbr = ["CHN", "PHI", "PIT", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]

    #This input will eventually be supplied by web scraper data
    team1, team2 = input("Enter a team matchup:").split()
    print("Team 1: ",team1)
    print("Team 2: ", team2)

    data = pd.read_csv(team1 + '_All.csv')
    data.plot(kind="scatter", x="Home Team OBP", y="Home Team Score")
    plt.show()
    data["Home Team HBP"].hist()
    plt.show()

    # data.drop(['Visting Team', 'League', 'Home Team', 'League.1', 'Park ID'], axis=1)
    # data.drop(['Winning Pitcher ID', 'Losing Pitcher ID', 'Saving Pitcher ID', 'Visiting Starter Pitcher ID',
    # 'Home Starter Pitcher ID'], axis=1)
    # print(data.info())

    data.drop(['Visting Team','Date', 'League', 'Home Team', 'League.1', 'Park ID'], axis=1, inplace=True)
    data.drop(['Winning Pitcher ID', 'Visting Team Stolen Bases', 'Saving Pitcher ID','Home Team Stolen Bases', 'Visting Team Caught Stealing',
                'Home Team Caught Stealing', 'Visting Team G Double Play', 'Home Team G Double Play',
                'Losing Pitcher ID', 'Saving Pitcher ID', 'Visiting Starter Pitcher ID', 'Home Starter Pitcher ID',
                'Visting Team Awarded First on Interference', 'Home Team Awarded First on Interference',
                'Visting Team Balks', 'Home Team Balks', 'Visting Team Put-outs', 'Home Team Put-outs',
                'Visting Team Assists', 'Visting Team Passed Balls', 'Home Team Passed Balls',
                'Visting Team Double Plays','Attendance', 'Home Team Double Plays',
                'Home Team Triple Plays', 'Visting Team Triple Plays', 'Home Team Triples',
                'Visiting Team Sac Hits', 'Home Team Int Walks', 'Visting Team Int Walks',
                'Home Team Sac Hits', 'Length of Game in Outs', 'Visting Team Sac Flys', 'Home Team Sac Flys',
                'Home Team Wild Pitches', 'Home Team HBP', 'Visting Team HBP', 'Visting Team Wild Pitches',
                'Visiting Team Game Number', 'Home Team Game Number'], axis=1, inplace=True)

    try:
        data.drop(['Unnamed: 75'], axis=1, inplace=True)

    except KeyError:
        print("Column is not in the file!!!")


    # Using Pearson Correlation
    plt.subplots(figsize=(36,36))
    #plt.figure(figsize=(300, 100))
    cor = data.corr()
    sns.heatmap(cor, cmap=plt.cm.Reds)
    plt.show()
    plt.savefig("C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\Correlation_Heatmap.png")

    #X_train, X_test, y_train, y_test = train_test_split(data.drop('Home Team Score', axis=1),
                                                        #data['Home Team Score'], test_size=0.30,
                                                        #random_state=101)
    #linmodel = LinearRegression()
    #linmodel.fit(X_train,y_train)

    #linear_reg_score_train = linmodel.score(X_train, y_train)
    #print("Percentage correct on training set = ", 100. * linear_reg_score_train, "%")

    #predictions = linmodel.predict(X_test)
    #print("Predictions: ", predictions)


    # print(data.info())
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Win', axis=1),
                                                        data['Win'], test_size=0.30,
                                                        random_state=101)

    #print(data.isna().any())

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

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    plt.savefig("C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\Initial_Heatmap_WAS.png")


    rfe = RFE(logmodel, 9)  # running RFE
    rfe = rfe.fit(X_train, y_train)
    print(rfe.support_)  # Printing the boolean results
    print(rfe.ranking_)

    forest = ExtraTreesClassifier(n_estimators=500)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature Ranking:")

    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))


if __name__ == "__main__":
    main()