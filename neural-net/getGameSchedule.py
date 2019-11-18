import statsapi as mlb
import pandas as pd
from collections import namedtuple
from calendar import monthrange
import os

def main():
    year = 2019
    #getSchedule(year)
    #getGamePKs(year)
    teamAbbr = ["CHN", "PHI", "PIT", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]
    # Explore each variable in dataset
    for x in range(30):
        getGameData(teamAbbr[x],year)


def getSchedule(year):
    x = 1
    os.chdir('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\games')
    for month in range(4, 11):  # Month is always 1..12
        for day in range(1, monthrange(year, month)[1] + 1):

            date = str(month)+"/"+str(day)+"/"+str(year)
            games = mlb.schedule(start_date=date, end_date=date)
            print(games)

            schedule = pd.DataFrame(columns=['Home Team','Away Team', 'Game Time'])
            for game in games:
                homeTeam = game['home_name']
                if(homeTeam == "American League All-Stars" or homeTeam == "National League All-Stars"):
                    homeTeam = "New York Yankees"
                homeTeam = convertName(homeTeam)
                awayTeam = game['away_name']
                if (awayTeam == "American League All-Stars" or awayTeam == "National League All-Stars"):
                    awayTeam = "New York Mets"
                awayTeam = convertName(awayTeam)
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
                    #setting = "PM"
                    hours -= 12

                monthName = convertMonth(month)
                schedule.loc[x] = [homeTeam,awayTeam,monthName+" "+str(day)+" @ "+str(hours)+":"+minutes+setting]
                x = x+1

            schedule.to_csv('games_'+str(month)+'_'+str(day)+'_'+str(year)+'.csv', index=False)
            x=1


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

