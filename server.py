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
    games = pd.read_csv('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\games_9_22_2019.csv')
    data.drop(['Win', 'Team Abbr'], axis=1, inplace=True)
    predictions = model.predict(data)
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
    labels = (predictions < 0.5).astype(np.int32)
    print(labels)
    for x in range(games.shape[0]):
        awayTeamPrediction = 1
        awayTeamPrediction ^= labels[x]
        values = [games.iloc[x,2], games.iloc[x,0], int(labels[x]),"static/" +games.iloc[x,0]+"_Logo.png", games.iloc[x,1], int(awayTeamPrediction),"static/" +games.iloc[x,1]+"_Logo.png"]
        gameData.update(dict(zip(keys,values)))
        gameDataFinal.append(gameData)

    print(gameDataFinal)
    return gameDataFinal

if __name__ == '__main__':
    app.run()

