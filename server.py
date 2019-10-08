import numpy as np
from flask import Flask, request, render_template
import pickle
import numpy as np
from flask import request
import pandas as pd
from sklearn import preprocessing
import json

app = Flask(__name__)


@app.route('/')
def home():
    model = pickle.load(open('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\data-scraper\\logmodel.pkl', 'rb'))
    data = pd.read_csv('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\data-scraper\\team_averages.csv')
    games = pd.read_csv('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\games_3_28_2019.csv')
    df = modifyDF(data,games)
    df.drop(['Win', 'teamAbbr'], axis=1, inplace=True)
    predictions = model.predict(df)
    print(predictions)
    gameData=createJSON(games, predictions)
    return render_template('result.html',games=gameData)

#@app.route('/predict',methods=['POST'])
#def get_win():

#    model = pickle.load(open('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\data-scraper\\logmodel.pkl','rb'))
#    data = pd.read_csv('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\data-scraper\\team_averages.csv')
#    data.drop('Win', 'Team Abbr', axis=1)
#    predictions = model.predict(data)
#    print(predictions)
#    return render_template('result.html',prediction = predictions)

def createJSON(games,predictions):
    gameDataFinal = []
    gameData = {}
    keys = ["Time", "HomeTeamAbbr","HomePrediction","HomeLogoPath", "AwayTeamAbbr","AwayPrediction","AwayLogoPath"]
    labels = predictions.astype(np.int32)
    print(labels)
    for x in range(games.shape[0]):

        awayTeamPrediction = 1
        awayTeamPrediction ^= labels[x]
        values = [games.iloc[x,2], games.iloc[x,0], int(labels[x]),"static/" +games.iloc[x,0]+"_Logo.png", games.iloc[x,1], int(awayTeamPrediction),"static/" +games.iloc[x,1]+"_Logo.png"]
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
                df.loc[idx + 1] = [row2['Team Abbr'], row2['Score'], row2['isHomeTeam'], row2['atBats'],
                                 row2['Hits'], row2['Doubles'], row2['Triples'], row2['homeRuns'], row2['RBI'],
                                 row2['Walks'], row2['Strikeouts'], row2['LOB'], row2['pitchersUsed'], row2['indER'],
                                 row2['teamER'], row2['Errors'], row2['battingAverage'], row2['OBP'], row2['Slugging'],
                                 row2['OPS'], row2['Win']]

    print(df)
    return df

if __name__ == '__main__':
    app.run()

