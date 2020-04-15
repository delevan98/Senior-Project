import brScraper
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import re
import os
import datetime
import pickle
import tensorflow as tf
import sqlite3
import dataCleaning
import server
import numpy as np
import statModels
import neuralNetwork



teamAbbr = ["CHC","PHI","PIT", "CIN", "STL", "BOS", "CHW",
				"CLE", "DET", "NYY", "BAL", "LAD", "SFG", "MIN",
			    "HOU", "NYM", "ATL", "OAK", "KCR", "SDP", "TEX",
				"TOR", "SEA", "MIA", "COL", "LAA", "TBR", "ARI",
				"MIL", "WSN"]

def main():
   scraper = brScraper.BaseballScraper()

   year = 2019

   #iterate 3/28/2019 through 9/29/2019

   start_date = datetime.date(2019, 7, 8)
   end_date = datetime.date(2019, 9, 29)
   delta = datetime.timedelta(days=1)

   while start_date <= end_date:
        print(str(start_date))
        # Load each model in
        print('Loading each model!')
        logModel = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\logmodel.pkl', 'rb'))

        linReg = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\linmodel.pkl', 'rb'))
        nnReg = tf.keras.models.load_model('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\regModel')
        rfReg = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\rflin.pkl', 'rb'))
        xgbReg = pickle.load(open('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\xgblin.pkl', 'rb'))
        print('Loaded each model!')

        # Read in data
        print('Loading team averages')
        data = pd.read_csv('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\team_averages.csv')
        print('Average file loaded')

        #Get games for specified date
        print('Getting games for ' + str(start_date))
        connection = sqlite3.connect("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\games\\gamesSchedule.db")
        crsr = connection.cursor()

        sql_command = "SELECT game_id, homeTeam, awayTeam FROM games WHERE gamedate = " + "'" + str(start_date) + "'"

        print('Acquired games for ' + str(start_date))

        # Convert database games to a dataframe
        print('Converting database games to dataframe')
        games = pd.DataFrame(crsr.execute(sql_command), columns=['game_id','Home Team', 'Away Team'])

        connection.close()
        print('Converted database games to dataframe')

        if(games.empty):
            start_date += delta

        else:
            # Reduce games dataframe to the rows for the home team
            print('Starting log model predictions')
            logDF = server.modifyDF(data, games)
            logDF.drop(['Win', 'teamAbbr'], axis=1, inplace=True)

            # Use logModel to predict the winner of a game

            winPredictions = logModel.predict(logDF)
            print('Finished log model predctions')

            # Reload team_averages.csv and rearrange data so that it reflects the order of games in game dataframe
            print('Starting regression model predictions')
            data = pd.read_csv('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\team_averages.csv')
            linearDF = server.modifyLinear(data, games)
            linearDF.drop(['teamAbbr', 'Win', 'wonPrev', 'OPS'], axis=1, inplace=True)

            # Predict the scores for both the home team and away team
            linRegScorePredictions = linReg.predict(linearDF)
            nnRegScorePredictions = nnReg.predict(linearDF)
            rfRegScorePredictions = rfReg.predict(linearDF)
            xgbRegScorePredictions = xgbReg.predict(linearDF)

            print('Finished regression model predictions')

            # Update the database with these predictions
            print("Updating database with predictions")

            labels = winPredictions.astype(np.int64)
            for x in range(games.shape[0]):
               awayTeamPrediction = 1
               awayTeamPrediction ^= labels[x]
               homeTeamPrediction = labels[x]

               linHomeScorePrediction = int(np.round(linRegScorePredictions[x * 2]))
               linAwayScorePrediction = int(np.round(linRegScorePredictions[2 * x + 1]))

               nnHomeScorePrediction = int(np.round(nnRegScorePredictions[x * 2]))
               nnAwayScorePrediction = int(np.round(nnRegScorePredictions[2 * x + 1]))

               rfHomeScorePrediction = int(np.round(rfRegScorePredictions[x * 2]))
               rfAwayScorePrediction = int(np.round(rfRegScorePredictions[2 * x + 1]))

               xgbHomeScorePrediction = int(np.round(xgbRegScorePredictions[x * 2]))
               xgbAwayScorePrediction = int(np.round(xgbRegScorePredictions[2 * x + 1]))

               insertVariblesIntoTable(int(games.iloc[x,0]), int(awayTeamPrediction), int(homeTeamPrediction),
                                       linHomeScorePrediction, linAwayScorePrediction,
                                       nnHomeScorePrediction, nnAwayScorePrediction,
                                       rfHomeScorePrediction, rfAwayScorePrediction,
                                       xgbHomeScorePrediction, xgbAwayScorePrediction)

            averages = pd.DataFrame(columns=['Score', 'isHomeTeam', 'atBats', 'Hits', 'Doubles', 'Triples',
                                            'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                            'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP',
                                            'Slugging', 'OPS', 'Win', 'wonPrev', 'WHIP', 'KPercent', 'BBPercent',
                                            'FIP', 'BABIP', 'ERA', 'HAllowed', 'defensiveSO'])
            for team in teamAbbr:
               teamName = team
               print('Getting new daily data for ' + teamName)

               battingData = scraper.parse(page_url="teams/tgl.cgi?team=" + teamName + "&t=b&year=" + str(year),
                                           date=str(start_date))

               if(battingData.empty):
                   print('Updating averages file even if there is no new data')
                   convertedTeamAbbr = brScraper.convertAbbr(teamName)
                   averages = averages.append(dataCleaning.write_averages(convertedTeamAbbr))
                   print('Done Updating averages file even if there was no new data')

               else:
                   time.sleep(5)
                   pitchingData = scraper.parse(page_url="teams/tgl.cgi?team=" + teamName + "&t=p&year=" + str(year),
                                                date=str(start_date))

                   dfWithPitching = pd.concat([battingData, pitchingData], axis=1)
                   dfWithPitching.drop(['date_game', 'PA'], axis=1, inplace=True)
                   dfWithPitching.to_csv("singleRow.csv", index=False)
                   # dfWithPitching.to_csv(teamName + str(x) + ".csv", index=False)

                   fullDF = pd.DataFrame(columns=['teamAbbr', 'League', 'Score', 'isHomeTeam', 'atBats', 'Hits',
                                                  'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                                  'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP',
                                                  'Slugging', 'OPS', 'Win', 'wonPrev', 'WHIP', 'KPercent', 'BBPercent',
                                                  'FIP', 'BABIP', 'ERA', 'HAllowed', 'defensiveSO'])

                   fullDF = scraper.cleanData(fullDF, teamName)

                   #fullDF.to_csv('cleanedRow.csv', index=False)
                   print('Got new data for ' + teamName)

                   # append to _Full.csv file
                   print('Appending new data to file')
                   convertedTeamAbbr = brScraper.convertAbbr(teamName)
                   teamDF = pd.read_csv(convertedTeamAbbr + '_Full.csv')

                   teamDF = teamDF.append(fullDF)
                   teamDF.to_csv(convertedTeamAbbr + '_Full.csv', index=False)

                   print('Appended new data to file')

                   # Update team_averages.csv
                   print('Updating averages file')
                   averages = averages.append(dataCleaning.write_averages(convertedTeamAbbr))
                   print('Done Updating averages file')

            # Save fully updates averages file
            print('Saving new averages file to disk')
            os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
            averages.to_csv('team_averages.csv', index=False)
            print('Saved new averages file to disk')

            # Append the new _Full.csv files into the combinedData.csv file
            print('Append updated files to a combined file')
            frames = []
            for x in range(30):
               frames.append(pd.read_csv(brScraper.convertAbbr(teamAbbr[x]) + '_Full.csv'))

            result = pd.concat(frames)
            result['wonPrev'].fillna(0, inplace=True)
            result.to_csv('combinedData.csv', index=False)

            print('Appended updated files to a combined file')

            # Re-train/save all the models using their respective functions
            print('Re-training statistical models')
            statModels.main()
            print('Re-trained statistical models')

            print('Re-training neural network models')
            #os.remove("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\weights\\Weights-best.hdf5")
            neuralNetwork.main()
            print('Re-trained neural network models')

            start_date += delta


def insertVariblesIntoTable(game_id, awayTeamPred, homeTeamPred, linHomeScorePred, linAwayScorePred, nnHomeScorePred, nnAwayScorePred, rfHomeScorePred, rfAwayScorePred, xgbHomeScorePred, xgbAwayScorePred):
    try:
        connection = sqlite3.connect("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\games\\gamesSchedule.db")

        crsr = connection.cursor()

        sql_command = "UPDATE games " \
                      "SET linPredHomeScore = ?, linPredAwayScore = ?, nnPredHomeScore = ?, nnPredAwayScore = ?," \
                      "  rfPredHomeScore = ?, rfPredAwayScore = ?, xgbPredHomeScore = ?, xgbPredAwayScore = ?," \
                      "  logHomeWinPred = ?, logAwayWinPred = ?" \
                      " WHERE game_id = ?;"

        recordTuple = (linHomeScorePred, linAwayScorePred, nnHomeScorePred, nnAwayScorePred, rfHomeScorePred,
                       rfAwayScorePred, xgbHomeScorePred, xgbAwayScorePred, homeTeamPred, awayTeamPred, game_id)

        crsr.execute(sql_command, recordTuple)

        connection.commit()
        print("Record inserted successfully into games table")

    except connection.Error as error:
        print("Failed to insert into SQLite table {}".format(error))

    finally:
        crsr.close()
        connection.close()
        print("Database successfully updated with predictions")

if __name__ == "__main__":
    main()