import numpy as np
from flask import Flask, request, render_template
import pickle
import numpy as np
from flask import request
import pandas as pd
from sklearn import preprocessing
import json


teamAbbr = ["CHN","PHI","PIT", "CIN", "SLN", "BOS", "CHA",
				"CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
			    "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
				"TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
				"MIL", "WAS"]

app = Flask(__name__)


@app.route('/')
def home():
    logModel = pickle.load(open('app/data-scraper/logmodel.pkl', 'rb'))
    linModel = pickle.load(open('app/data-scraper/linmodel.pkl', 'rb'))
    data = pd.read_csv('app/data-scraper/team_averages.csv')
    games = pd.read_csv('app/games/games_3_28_2019.csv')
    logDF = modifyDF(data,games)
    logDF.drop(['Win', 'teamAbbr'], axis=1, inplace=True)
    winPredictions = logModel.predict(logDF)
    print(winPredictions)

    linearDF = modifyLinear(data,games)
    linearDF.drop(['Win','Score','RBI', 'teamAbbr'], axis=1, inplace=True)
    scorePredictions = linModel.predict(linearDF)
    print(np.round(scorePredictions))
    gameData=createJSON(games, winPredictions, scorePredictions)
    return render_template('result.html',games=gameData)

@app.route('/predict',methods=['GET','POST'])
def get_win():
    logModel = pickle.load(open('app/data-scraper/logmodel.pkl', 'rb'))
    linModel = pickle.load(open('app/data-scraper/linmodel.pkl', 'rb'))
    data = pd.read_csv('app/data-scraper/team_averages.csv')

    matchupData = []

    if request.method == "POST":
        awayAbbr = request.form.get("Away Team")
        homeAbbr = request.form.get("Home Team")

        matchup = pd.DataFrame(columns=['Home Team', 'Away Team', 'Game Time'])
        matchup.loc[1] = [homeAbbr, awayAbbr, "September 21, 2019"]
        logDF = modifyDF(data,matchup)
        logDF.drop(['Win', 'teamAbbr'], axis=1, inplace=True)
        winPredictions = logModel.predict(logDF)

        linearDF = modifyLinear(data, matchup)
        linearDF.drop(['Win', 'Score', 'RBI', 'teamAbbr'], axis=1, inplace=True)
        scorePredictions = linModel.predict(linearDF)

        matchupData = createJSON(matchup, winPredictions, scorePredictions)

    return render_template('matchup.html',teams = teamAbbr, predictions=matchupData)

def createJSON(games,predictions,scores):
    gameDataFinal = []
    gameData = {}
    keys = ["Time", "HomeTeamAbbr","HomePrediction","HomeLogoPath","HomeScore", "AwayTeamAbbr","AwayPrediction","AwayLogoPath", "AwayScore"]
    labels = predictions.astype(np.int32)
    print(labels)
    for x in range(games.shape[0]):

        awayTeamPrediction = 1
        awayTeamPrediction ^= labels[x]
        values = [games.iloc[x,2], games.iloc[x,0], int(labels[x]),"static/" +games.iloc[x,0]+"_Logo.png",np.int32(np.floor(scores[x*2])), games.iloc[x,1], int(awayTeamPrediction),"static/" +games.iloc[x,1]+"_Logo.png", np.int32(np.floor(scores[2*x+1]))]
        print(values)
        gameData = dict(zip(keys,values))
        gameDataFinal.append(gameData)

    print(gameDataFinal)
    return gameDataFinal

def modifyDF(data,games):
    df = pd.DataFrame(columns=['teamAbbr', 'Score', 'isHomeTeam', 'atBats', 'Hits',
                                  'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                  'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                                  'OPS', 'Win'])

    for (idx, row) in games.iterrows():
        for (idx2, row2) in data.iterrows():
            # Set is home team to 1 and wonPrev to its value as it is an average in this row
            if(row2.loc['Team Abbr'] == row.loc['Home Team']):
                df.loc[idx + 1] = [row2['Team Abbr'], row2['Score'], 1, row2['atBats'],
                                 row2['Hits'], row2['Doubles'], row2['Triples'], row2['homeRuns'], row2['RBI'],
                                 row2['Walks'], row2['Strikeouts'], row2['LOB'], row2['pitchersUsed'], row2['indER'],
                                 row2['teamER'], row2['Errors'], row2['battingAverage'], row2['OBP'], row2['Slugging'],
                                 row2['OPS'], row2['Win']]

    print(df.tail(10))
    return df

def modifyLinear(data,games):
    df = pd.DataFrame(columns=['teamAbbr', 'Score', 'isHomeTeam', 'atBats', 'Hits',
                               'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                               'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                               'OPS', 'Win'])
    x = 0
    for (idx, row) in games.iterrows():
        for (idx2,row2) in data.iterrows():
            # Set is home team to 1 and wonPrev to its value as it is an average in this row
            if(row.loc['Home Team'] == row2.loc['Team Abbr']):
                df.loc[x+1] = [row2['Team Abbr'], row2['Score'], 1, row2['atBats'],
                                 row2['Hits'], row2['Doubles'], row2['Triples'], row2['homeRuns'], row2['RBI'],
                                 row2['Walks'], row2['Strikeouts'], row2['LOB'], row2['pitchersUsed'], row2['indER'],
                                 row2['teamER'], row2['Errors'], row2['battingAverage'], row2['OBP'], row2['Slugging'],
                                 row2['OPS'], row2['Win']]
                x = x + 1

        for (idx3,row3) in data.iterrows():
            if(row.loc['Away Team'] == row3.loc['Team Abbr']):
                df.loc[x+1] = [row3['Team Abbr'], row3['Score'], 0, row3['atBats'],
                                   row3['Hits'], row3['Doubles'], row3['Triples'], row3['homeRuns'], row3['RBI'],
                                   row3['Walks'], row3['Strikeouts'], row3['LOB'], row3['pitchersUsed'], row3['indER'],
                                   row3['teamER'], row3['Errors'], row3['battingAverage'], row3['OBP'],
                                   row3['Slugging'], row3['OPS'], row3['Win']]
                x = x + 1

    print(df.tail(10))
    return df
if __name__ == '__main__':
    app.run()

