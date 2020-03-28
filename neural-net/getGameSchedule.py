import statsapi as mlb
import pandas as pd
from collections import namedtuple
from calendar import monthrange
import os
import sqlite3

def main():
    year = 2019
    getSchedule(2019)

def getSchedule(year):
    x = 1

    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\games')

    for month in range(4, 11):
        for day in range(1, monthrange(year, month)[1] + 1):

            date = str(month)+"/"+str(day)+"/"+str(year)
            games = mlb.schedule(start_date=date, end_date=date)
            print(games)

            schedule = pd.DataFrame(columns=['Home Team','Away Team', 'Game Date', 'Game Time'])
            for game in games:
                game_id = game['game_id']
                homeTeamAbbr = game['home_name']
                if(homeTeamAbbr == "American League All-Stars" or homeTeamAbbr == "National League All-Stars"):
                    homeTeamAbbr = "New York Yankees"
                homeTeamAbbr = convertName(homeTeamAbbr)
                awayTeamAbbr = game['away_name']
                if (awayTeamAbbr == "American League All-Stars" or awayTeamAbbr == "National League All-Stars"):
                    awayTeamAbbr = "New York Mets"
                awayTeamAbbr = convertName(awayTeamAbbr)
                time = game['game_datetime']

                scheduledTime = time[time.find('T')+1:time.find('Z')]
                splitTime = scheduledTime.split(":")
                hours = int(splitTime[0])
                hours = hours - 4
                if(hours < 0):
                    hours = hours + 12
                minutes = splitTime[1]
                setting = "PM"
                if hours > 12:
                    hours -= 12

                monthName = convertMonth(month)
                gameTime = str(hours) + ":"+ minutes + setting
                homeActualScore = game['home_score']
                awayActualScore = game['away_score']

                actualHomeWin = 0
                if(homeActualScore > awayActualScore):
                    actualHomeWin = 1

                insertVariblesIntoTable(game_id, homeTeamAbbr, awayTeamAbbr, game['game_date'], gameTime, homeActualScore, awayActualScore, actualHomeWin)
                #schedule.loc[x] = [homeTeam,awayTeam,monthName+" "+str(day)+" @ "+str(hours)+":"+minutes+setting]
                x = x+1

            #schedule.to_csv('games_'+str(month)+'_'+str(day)+'_'+str(year)+'.csv', index=False)
            x=1

def insertVariblesIntoTable(game_id, homeTeam, awayTeam, gameDate, gameTime, homeActualScore, awayActualScore, actualHomeWin):
    try:
        connection = sqlite3.connect("gamesSchedule.db")
        crsr = connection.cursor()
        sql_command = "INSERT INTO games (game_id, homeTeam, awayTeam, gameDate, gameTime," \
                      "  actualHomeScore, actualAwayScore, actualHomeWin, predHomeScore, predAwayScore, " \
                      " isHomeWinPredicted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL);"

        recordTuple = (game_id, homeTeam, awayTeam, gameDate, gameTime, homeActualScore, awayActualScore, actualHomeWin)
        crsr.execute(sql_command, recordTuple)
        connection.commit()
        print("Record inserted successfully into games table")

    except connection.Error as error:
        print("Failed to insert into MySQL table {}".format(error))

    finally:
        crsr.close()
        connection.close()
        print("SQLite connection is closed")

def convertName(team):
    teamNames = {"Arizona Diamondbacks": "ARI",
                 "Atlanta Braves": "ATL",
                 "Baltimore Orioles": "BAL",
                 "Boston Red Sox": "BOS",
                 "Chicago White Sox": "CHA",
                 "Chicago Cubs": "CHN",
                 "Cincinnati Reds": "CIN",
                 "Cleveland Indians": "CLE",
                 "Colorado Rockies": "COL",
                 "Detroit Tigers": "DET",
                 "Houston Astros": "HOU",
                 "Kansas City Royals": "KCA",
                 "Los Angeles Angels": "ANA",
                 "Los Angeles Dodgers": "LAN",
                 "Miami Marlins": "FLO",
                 "Milwaukee Brewers": "MIL",
                 "Minnesota Twins": "MIN",
                 "New York Yankees": "NYA",
                 "New York Mets": "NYN",
                 "Oakland Athletics": "OAK",
                 "Philadelphia Phillies": "PHI",
                 "Pittsburgh Pirates": "PIT",
                 "San Diego Padres": "SDN",
                 "San Francisco Giants": "SFN",
                 "Seattle Mariners": "SEA",
                 "St. Louis Cardinals": "SLN",
                 "Tampa Bay Rays": "TBA",
                 "Texas Rangers": "TEX",
                 "Toronto Blue Jays": "TOR",
                 "Washington Nationals": "WAS"

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


def formatMonth(month):
    if(month >= 1 and month < 10):
        month = str(0) + str(month)

    return month


def getGamePKs(year):
    x = 1
    os.chdir('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\games')
    for month in range(4, 11):  # Month is always 1..12
        for day in range(1, monthrange(year, month)[1] + 1):
            date = str(month) + "/" + str(day) + "/" + str(year)
            games = mlb.schedule(start_date=date, end_date=date)

            for game in games:
                game_id = game['game_id']
                print(mlb.boxscore_data(game_id))
                x = x + 1


def getGameData(teamAbbr,year):
    url = "http://www.baseball-reference.com/teams/tgl.cgi?team=" + teamAbbr + "&t=b&year=" + str(year)

if __name__ == "__main__":
    main()

