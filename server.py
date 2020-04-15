from flask import Flask, request, render_template
import pickle
import numpy as np
from flask import request
import pandas as pd
import tensorflow as tf
from datetime import date, timedelta, datetime
import sqlite3
import re


teamAbbr = ["CHN","PHI","PIT", "CIN", "SLN", "BOS", "CHA",
				"CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
			    "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
				"TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
				"MIL", "WAS"]

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route('/')
@app.route("/<day>")
def home(day=str(date.today())):
    date_object = datetime.strptime(day, '%Y-%m-%d')
    print(str(date_object))
    next_date_object = date_object + timedelta(days=1)
    next_day = datetime.strftime(next_date_object, '%Y-%m-%d')
    previous_date_object = date_object - timedelta(days=1)
    previous_day = datetime.strftime(previous_date_object, '%Y-%m-%d')

    # logModel = pickle.load(open('/app/data-scraper/logmodel.pkl', 'rb'))
    # linModel = pickle.load(open('/app/data-scraper/linmodel.pkl', 'rb'))
    # linModel = tf.keras.models.load_model('/app/neural-net/models/regModel')
    # data = pd.read_csv('/app/data-scraper/team_averages.csv')
    # games = pd.read_csv('/app/games/games_3_28_2019.csv')

    # logModel = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\logmodel.pkl', 'rb'))
    # linModel = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\linmodel.pkl', 'rb'))
    # linModel = tf.keras.models.load_model('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\regModel')
    # data = pd.read_csv('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\team_averages.csv')
    # games = pd.read_csv('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\games\\games_3_28_2019.csv')

    # logDF = modifyDF(data,games)
    # logDF.drop(['Win', 'teamAbbr'], axis=1, inplace=True)
    # winPredictions = logModel.predict(logDF)
    # print(winPredictions)

    # linearDF = modifyLinear(data,games)
    # linearDF.drop(['teamAbbr', 'Win'], axis=1, inplace=True)
    # print(linearDF.dtypes)
    # scorePredictions = linModel.predict(linearDF)
    # print(np.round(scorePredictions))

    #connection = sqlite3.connect("app/games/gamesSchedule.db")
    connection = sqlite3.connect("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\games\\gamesSchedule.db")
    crsr = connection.cursor()
    print(day)

    sql_command = "SELECT homeTeam, awayTeam, gameDate, gameTime," \
                  " xgbPredHomeScore, xgbPredAwayScore, logHomeWinPred, logAwayWinPred FROM games WHERE gamedate = " + "'" + str(day) + "'"
    games = pd.DataFrame(crsr.execute(sql_command), columns=['Home Team', 'Away Team', 'gameDate', 'gameTime',
                                                             'xgbPredHomeScore', 'xgbPredAwayScore',
                                                             'logHomeWinPrediction', 'logAwayWinPrediction'])

    gameData=createJSON(games, mode=1)
    return render_template('result.html',games=gameData, day=day, previous_day=previous_day, next_day=next_day)

@app.route('/predict',methods=['GET','POST'])
def get_win():
    logModel = pickle.load(open('/app/data-scraper/logmodel.pkl', 'rb'))
    linModel = tf.keras.models.load_model('/app/neural-net/models/regModel')
    data = pd.read_csv('/app/data-scraper/team_averages.csv')

    #logModel = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\logmodel.pkl', 'rb'))
    #linModel = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\linmodel.pkl', 'rb'))
    #linModel = tf.keras.models.load_model('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\regModel')
    #data = pd.read_csv('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\team_averages.csv')
    teamNames = []
    matchupData = []

    for x in range(30):
        abbreviation = teamAbbr[x]
        teamNames.append(convertName(abbreviation))

    if request.method == "POST":
        awayName = request.form.get("Away Team")
        homeName = request.form.get("Home Team")

        awayAbbr = convertAbbr(awayName)
        homeAbbr = convertAbbr(homeName)


        matchup = pd.DataFrame(columns=['Home Team', 'Away Team', 'Game Time'])
        matchup.loc[1] = [homeAbbr, awayAbbr, "September 21, 2019"]
        logDF = modifyDF(data,matchup)
        logDF.drop(['Win', 'teamAbbr'], axis=1, inplace=True)
        winPredictions = logModel.predict(logDF)

        linearDF = modifyLinear(data, matchup)
        linearDF.drop(['Win', 'teamAbbr'], axis=1, inplace=True)
        print(linearDF.dtypes)
        scorePredictions = linModel.predict(linearDF)

        matchupData = createJSON(matchup, winPredictions, scorePredictions, mode=2)

    return render_template('matchup.html',teams = teamNames, predictions=matchupData)

def createJSON(games, predictions=None, scores=None, mode=1):
    gameDataFinal = []
    gameData = {}
    keys = ["Time", "HomeTeamAbbr","HomePrediction","HomeLogoPath","HomeScore", "AwayTeamAbbr","AwayPrediction","AwayLogoPath", "AwayScore"]
    #labels = predictions.astype(np.int32)

    if (mode == 1):
        for x in range(games.shape[0]):

            homeTeamName = convertName(games.iloc[x, 0])
            awayTeamName = convertName(games.iloc[x, 1])

            gameDate = games.iloc[x, 2]

            gameTime = games.iloc[x, 3]

            homeScorePrediction = games.iloc[x, 4]
            awayScorePrediction = games.iloc[x, 5]

            homeTeamPrediction = games.iloc[x, 6]
            awayTeamPrediction = games.iloc[x, 7]

            # Checks and balances
            # Tie Predicted
            if((homeScorePrediction == awayScorePrediction) and homeTeamPrediction == 1):
                homeScorePrediction = homeScorePrediction + 1
            # Tie Predicted
            elif((homeScorePrediction == awayScorePrediction) and homeTeamPrediction == 0):
                awayScorePrediction = awayScorePrediction + 1

            elif((homeTeamPrediction == 1) and homeScorePrediction < awayScorePrediction):
                homeTeamPrediction = 0
                awayTeamPrediction = 1

            elif((awayTeamPrediction == 1) and awayScorePrediction < homeScorePrediction):
                awayTeamPrediction = 0
                homeTeamPrediction = 1

            regex = re.compile('\d+-(\d{2})-\d+')
            month = regex.findall(gameDate)

            month = convertMonth(int(month[0].lstrip('0')))

            regex = re.compile('\d+-\d{2}-(\d{2})')
            day = regex.findall(gameDate)

            date = day[0].lstrip('0')

            gameDate = month + " " + date + " @ " + gameTime

            values = [gameDate, homeTeamName, int(homeTeamPrediction), "static/" + games.iloc[x, 0] + "_Logo.png",
                      int(homeScorePrediction), awayTeamName, int(awayTeamPrediction),
                      "static/" + games.iloc[x, 1] + "_Logo.png", int(awayScorePrediction)]

            gameData = dict(zip(keys, values))
            gameDataFinal.append(gameData)

    elif(mode == 2):
        labels = predictions.astype(np.int32)

        for x in range(games.shape[0]):

            awayTeamPrediction = 1
            awayTeamPrediction ^= labels[x]
            homeTeamPrediction = labels[x]
            homeTeamName = convertName(games.iloc[x, 0])
            awayTeamName = convertName(games.iloc[x, 1])
            homeScorePrediction = np.int32(np.floor(scores[x * 2]))
            awayScorePrediction = np.int32(np.floor(scores[2 * x + 1]))

            # Checks and balances
            # Tie Predicted
            # Make this a method
            if ((homeScorePrediction == awayScorePrediction) and labels[x] == 1):
                homeScorePrediction = homeScorePrediction + 1
            # Tie Predicted
            elif ((homeScorePrediction == awayScorePrediction) and labels[x] == 0):
                awayScorePrediction = awayScorePrediction + 1

            elif ((homeTeamPrediction == 1) and homeScorePrediction < awayScorePrediction):
                homeTeamPrediction = 0
                awayTeamPrediction = 1

            elif ((awayTeamPrediction == 1) and awayScorePrediction < homeScorePrediction):
                awayTeamPrediction = 0
                homeTeamPrediction = 1

        gameDate = games.iloc[x,2]

        values = [gameDate, homeTeamName, int(homeTeamPrediction),"static/" +games.iloc[x,0]+"_Logo.png",
                  int(homeScorePrediction), awayTeamName, int(awayTeamPrediction),
                  "static/" + games.iloc[x,1]+"_Logo.png", int(awayScorePrediction)]

        gameData = dict(zip(keys,values))
        gameDataFinal.append(gameData)

    return gameDataFinal

def modifyDF(data,games):
    df = pd.DataFrame(columns=['teamAbbr', 'Score', 'isHomeTeam', 'atBats', 'Hits',
                                  'Doubles', 'Triples', 'homeRuns', 'Walks', 'Strikeouts', 'LOB',
                                  'pitchersUsed', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                                  'OPS', 'Win', 'wonPrev', 'WHIP', 'KPercent', 'BBPercent', 'FIP', 'BABIP', 'ERA',
                                  'HAllowed', 'defensiveSO'])

    for (idx, row) in games.iterrows():
        for (idx2, row2) in data.iterrows():
            # Set is home team to 1 and wonPrev to its value as it is an average in this row
            if(row2.loc['TeamAbbr'] == row.loc['Home Team']):
                df.loc[idx + 1] = [row2['TeamAbbr'], row2['Score'], 1, row2['atBats'],
                                 row2['Hits'], row2['Doubles'], row2['Triples'], row2['homeRuns'],
                                 row2['Walks'], row2['Strikeouts'], row2['LOB'], row2['pitchersUsed'],
                                 row2['Errors'], row2['battingAverage'], row2['OBP'], row2['Slugging'],
                                 row2['OPS'], row2['Win'], row2['wonPrev'], row2['WHIP'], row2['KPercent'],
                                 row2['BBPercent'], row2['FIP'], row2['BABIP'], row2['ERA'], row2['HAllowed'],
                                 row2['defensiveSO']]

    #print(df.tail(10))
    return df

def modifyLinear(data,games):
    df = pd.DataFrame(columns=['teamAbbr', 'isHomeTeam', 'atBats', 'Hits',
                               'Doubles', 'Triples', 'homeRuns', 'Walks', 'Strikeouts', 'LOB',
                               'Errors', 'battingAverage', 'OBP', 'Slugging',
                               'OPS', 'Win', 'wonPrev', 'WHIP', 'KPercent', 'BBPercent', 'FIP', 'BABIP', 'ERA',
                               'HAllowed', 'defensiveSO'])
    x = 0
    for (idx, row) in games.iterrows():
        for (idx2,row2) in data.iterrows():
            # Set is home team to 1 and wonPrev to its value as it is an average in this row
            if(row.loc['Home Team'] == row2.loc['TeamAbbr']):
                df.loc[x+1] = [row2['TeamAbbr'], np.float64(1), row2['atBats'],
                                 row2['Hits'], row2['Doubles'], row2['Triples'], row2['homeRuns'],
                                 row2['Walks'], row2['Strikeouts'], row2['LOB'], row2['Errors'], row2['battingAverage'], row2['OBP'], row2['Slugging'],
                                 row2['OPS'], row2['Win'], row2['wonPrev'], row2['WHIP'], row2['KPercent'],
                                 row2['BBPercent'], row2['FIP'], row2['BABIP'], row2['ERA'], row2['HAllowed'], row2['defensiveSO']]
                x = x + 1

        for (idx3,row3) in data.iterrows():
            if(row.loc['Away Team'] == row3.loc['TeamAbbr']):
                df.loc[x+1] = [row3['TeamAbbr'], np.float64(0), row3['atBats'],
                                   row3['Hits'], row3['Doubles'], row3['Triples'], row3['homeRuns'],
                                   row3['Walks'], row3['Strikeouts'], row3['LOB'], row3['Errors'], row3['battingAverage'], row3['OBP'],
                                   row3['Slugging'], row3['OPS'], row3['Win'], row3['wonPrev'], row3['WHIP'], row3['KPercent'],
                                   row3['BBPercent'], row3['FIP'], row3['BABIP'], row3['ERA'], row3['HAllowed'], row3['defensiveSO']]
                x = x + 1

    #print(df.tail(10))
    return df

def convertName(teamAbbr):
    teamNames = {"ARI": "Diamondbacks",
                 "ATL": "Braves",
                 "BAL": "Orioles",
                 "BOS": "Red Sox",
                 "CHA": "White Sox",
                 "CHN": "Cubs",
                 "CIN": "Reds",
                 "CLE": "Indians",
                 "COL": "Rockies",
                 "DET": "Tigers",
                 "HOU": "Astros",
                 "KCA": "Royals",
                 "ANA": "Angels",
                 "LAN": "Dodgers",
                 "FLO": "Marlins",
                 "MIL": "Brewers",
                 "MIN": "Twins",
                 "NYA": "Yankees",
                 "NYN": "Mets",
                 "OAK": "Athletics",
                 "PHI": "Phillies",
                 "PIT": "Pirates",
                 "SDN": "Padres",
                 "SFN": "Giants",
                 "SEA": "Mariners",
                 "SLN": "Cardinals",
                 "TBA": "Rays",
                 "TEX": "Rangers",
                 "TOR": "Blue Jays",
                 "WAS": "Nationals"

    }

    convertedName = teamNames[teamAbbr]
    return convertedName

def convertAbbr(team):
    teamNames = {"Diamondbacks": "ARI",
                 "Braves": "ATL",
                 "Orioles": "BAL",
                 "Red Sox": "BOS",
                 "White Sox": "CHA",
                 "Cubs": "CHN",
                 "Reds": "CIN",
                 "Indians": "CLE",
                 "Rockies": "COL",
                 "Tigers": "DET",
                 "Astros": "HOU",
                 "Royals": "KCA",
                 "Angels": "ANA",
                 "Dodgers": "LAN",
                 "Marlins": "FLO",
                 "Brewers": "MIL",
                 "Twins": "MIN",
                 "Yankees": "NYA",
                 "Mets": "NYN",
                 "Athletics": "OAK",
                 "Phillies": "PHI",
                 "Pirates": "PIT",
                 "Padres": "SDN",
                 "Giants": "SFN",
                 "Mariners": "SEA",
                 "Cardinals": "SLN",
                 "Rays": "TBA",
                 "Rangers": "TEX",
                 "Blue Jays": "TOR",
                 "Nationals": "WAS"
    }

    convertedName = teamNames[team]
    return convertedName

def convertMonth(month):
    months = {1: "January",
              2: "February",
              3: "March",
              4: "April",
              5: "May",
              6: "June",
              7: "July",
              8: "August",
              9: "September",
              10: "October",
              11: "November",
              12: "December"}

    newMonth = months[month]
    return newMonth

if __name__ == '__main__':
    app.run()

