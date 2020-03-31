from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import re
import os

teamAbbr = ["CHC","PHI","PIT", "CIN", "STL", "BOS", "CHW",
				"CLE", "DET", "NYY", "BAL", "LAD", "SFG", "MIN",
			    "HOU", "NYM", "ATL", "OAK", "KCR", "SDP", "TEX",
				"TOR", "SEA", "MIA", "COL", "LAA", "TBR", "ARI",
				"MIL", "WSN"]
os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')

class BaseballScraper:

    def __init__(self, url="https://www.baseball-reference.com/"):
        self.url = url
# Will end up being a general class that will parse tables on a given page on baseball-reference
# Inputs will include the url of page to be scraped along with an optional date parameter which will
# be used to gather new daily data in the actual implementation of the system
# Will return the dataframe with the data
# Whether that data will be cleaned or not is up for grabs
    def parse(self, page_url, date=""):

        def determineTableID(page_url):
            regex = re.compile('[t]\W(.)')
            match = regex.findall(page_url)

            if (match[0] == 'b'):
                table_id = 'team_batting_gamelogs'
            elif (match[0] == 'p'):
                table_id = 'team_pitching_gamelogs'

            return table_id

        URL = self.url + page_url
        page = requests.get(URL)

        soup = BeautifulSoup(page.content, 'html.parser')

        results = soup.find(id=determineTableID(page_url))

        ## Get header names located with tag name "data-stat" ##

        headers = results.find("thead").find_all("th")
        header_names = []

        for header in headers:
            if header.get('data-stat') != 'ranker':
                header_names.append(str(header.get('data-stat')))

        df = pd.DataFrame(columns=header_names)
        gameData = {}

        ## Find parseable rows in the table and extract the info
        if(date != ""):
            tag = results.find("td", attrs={'csk':re.compile(date + '\.[A-Z]+[0-9]+')})
            row = tag.find_parent('tr')

            x=1
            data = row.find_all("td")
            data_list = []

            for stat in data:
                ## 1 denotes a home game, in b-r it is either nothing or an @ ##
                if stat.string == None:
                    data_list.append(1)
                else:
                    data_list.append(str(stat.get_text()))

            gameData = dict(zip(header_names, data_list))

            ## If the dictionary is not empty add it to the dataframe ##
            if bool(gameData) != False:
                df.loc[x] = gameData
                x = x + 1
            else:
                print("Found header in middle of table!")

            return df


        else:
            rows = results.find('tbody').find_all('tr')

            x = 1
            ## Need to figure out how to search for a date in a row and include only that dates data ##
            ## if else above may no longer be needed ^^^^ ##
            for row in rows:
               data = row.find_all("td")
               data_list = []

               for stat in data:
                   ## 1 denotes a home game, in b-r it is either nothing or an @ ##
                   if stat.string == None:
                       data_list.append(1)
                   else:
                       data_list.append(str(stat.get_text()))

               gameData = dict(zip(header_names, data_list))

               ## If the dictionary is not empty add it to the dataframe ##
               if bool(gameData) != False:
                    df.loc[x] = gameData
                    x = x + 1
               else:
                    print("Found header in middle of table!")


            return df

    def clean(self,team_df, dfType):
        if(dfType == 1):
            print('e')

## Run the scraper for each team and clean the data that is coming in  ##
## After this data has been cleaned into an acceptable format append   ##
## the team's respective _Full.csv file. If the dataframe is empty, do ##
## not append it to the _Full.csv file as no new data was pulled       ##
## Once this is done combined all team's _Full file into a combinedData##
## file for input into a machine learning model                        ##
## Execute this nightly in a cron job with the previous day as the date##
## input and do the steps above                                        ##

def convertAbbr(team):
    teamNames = {"ARI": "ARI",
                 "ATL": "ATL",
                 "BAL": "BAL",
                 "BOS": "BOS",
                 "CHW": "CHA",
                 "CHC": "CHN",
                 "CIN": "CIN",
                 "CLE": "CLE",
                 "COL": "COL",
                 "DET": "DET",
                 "HOU": "HOU",
                 "KCR": "KCA",
                 "LAA": "ANA",
                 "LAD": "LAN",
                 "MIA": "FLO",
                 "MIL": "MIL",
                 "MIN": "MIN",
                 "NYY": "NYA",
                 "NYM": "NYN",
                 "OAK": "OAK",
                 "PHI": "PHI",
                 "PIT": "PIT",
                 "SDP": "SDN",
                 "SFG": "SFN",
                 "SEA": "SEA",
                 "STL": "SLN",
                 "TBR": "TBA",
                 "TEX": "TEX",
                 "TOR": "TOR",
                 "WSN": "WAS"
    }

    convertedName = teamNames[team]
    return convertedName

def getLeague(team):
    teamNames = {"ARI": "NL",
                 "ATL": "NL",
                 "BAL": "AL",
                 "BOS": "AL",
                 "CHW": "AL",
                 "CHC": "NL",
                 "CIN": "NL",
                 "CLE": "AL",
                 "COL": "NL",
                 "DET": "AL",
                 "HOU": "AL",
                 "KCR": "AL",
                 "LAA": "AL",
                 "LAD": "NL",
                 "MIA": "NL",
                 "MIL": "NL",
                 "MIN": "AL",
                 "NYY": "AL",
                 "NYM": "NL",
                 "OAK": "AL",
                 "PHI": "NL",
                 "PIT": "NL",
                 "SDP": "NL",
                 "SFG": "NL",
                 "SEA": "AL",
                 "STL": "NL",
                 "TBR": "AL",
                 "TEX": "AL",
                 "TOR": "AL",
                 "WSN": "NL"
    }

    convertedName = teamNames[team]
    return convertedName

def cleanData(cleanDF, teamAbbr):
    # This DF has columns with duplicated column names
    # read_csv will rename these columns automatically

    dirtyDF = pd.read_csv('singleRow.csv')
    print(dirtyDF)
    league = getLeague(teamAbbr)
    teamAbbr = convertAbbr(teamAbbr)

    fullDF = pd.read_csv(teamAbbr+'_Full.csv')

    lastRow = fullDF.tail(1)

    wonPrev = lastRow.iloc[0]['Win']

    for (idx, row) in dirtyDF.iterrows():
        if(row['team_homeORaway'] == '@'):
            isHomeTeam = 0
        else:
            isHomeTeam = 1

        result = str(row['game_result'])
        if (result[0] == 'W'):
            homeWinOrNo = 1

        else:
            homeWinOrNo = 0

        whip = (row['BB.1'] + row['H.1']) / row['IP']
        KPercent = row['SO.1'] / row['batters_faced']

        BBPercent = row['BB.1'] / row['batters_faced']
        fip = (((13*row['HR.1'])+(3*(row['BB.1']+row['HBP.1']))-(2*row['SO.1']))/ row['IP']) + 3.214
        BABIP = (row['H.1'] - row['HR.1']) / (row['AB.1'] - row['HR.1'] - row['SO.1'] + row['SF.1'])
        # fullDF = pd.DataFrame(columns=['teamAbbr', 'League', 'Score', 'isHomeTeam', 'atBats', 'Hits',
        #                               'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
        #                               'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
        #                               'OPS', 'Win', 'wonPrev', 'WHIP', 'KPercent', 'BBPercent', 'FIP', 'BABIP', 'ERA',
        #                               'HAllowed', 'defensiveSO'])
        cleanDF.loc[idx+1] = [teamAbbr, league, row['R'], isHomeTeam, row['AB'], row['H'], row['2B'],
                              row['3B'], row['HR'], row['RBI'], row['BB'], row['SO'], row['LOB'],
                              row['pitchers_number'], row['ER'], row['ER'], row['UER'], row['batting_avg'],
                              row['onbase_perc'], row['slugging_perc'], row['onbase_plus_slugging'],
                              homeWinOrNo, wonPrev, whip, KPercent, BBPercent, fip, BABIP, row['earned_run_avg'],
                              row['H.1'], row['SO.1']]
    return cleanDF


############## Put this in a function, where date is the only parameter
scraper = BaseballScraper()

# Later: Get year through the date string passed in
year = 2019

for team in teamAbbr:

    teamName = team

    battingData = scraper.parse(page_url="teams/tgl.cgi?team=" + teamName + "&t=b&year=" + str(year), date='2019-03-28')
    time.sleep(10)
    pitchingData = scraper.parse(page_url="teams/tgl.cgi?team=" + teamName + "&t=p&year=" + str(year), date='2019-03-28')

    dfWithPitching = pd.concat([battingData, pitchingData], axis=1)
    print(dfWithPitching.dtypes)
    dfWithPitching.drop(['date_game', 'PA'], axis=1, inplace=True)
    dfWithPitching.to_csv("singleRow.csv", index=False)
    #dfWithPitching.to_csv(teamName + str(x) + ".csv", index=False)

    fullDF = pd.DataFrame(columns=['teamAbbr', 'League', 'Score', 'isHomeTeam', 'atBats', 'Hits',
                                   'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                   'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                                   'OPS', 'Win', 'wonPrev', 'WHIP',  'KPercent', 'BBPercent', 'FIP', 'BABIP', 'ERA',
                                   'HAllowed', 'defensiveSO'])

    fullDF = cleanData(fullDF, teamName)

    fullDF.to_csv('cleanedRow.csv', index=False)

    # append to _Full.csv file and move to next team.
#########################################
    exit(0)





#pitchingData = scraper.parse(page_url="teams/tgl.cgi?team=CIN&t=p&year=2010")
#pitchingData.to_csv("pitching.csv", index=False)

#df = pd.read_csv("CIN2010.csv")

#dfWithPitching = pd.concat([df, pitchingData], axis=1)

#dfWithPitching.to_csv('CIN2010.csv', index=False)
#print(battingData)
#battingData.to_csv('batting.csv', index=False)

#time.sleep(20)

#pitchingData = scraper.parse(page_url="teams/tgl.cgi?team=CHC&t=p&year=2019")
#print(pitchingData)
#pitchingData.to_csv('pitching.csv', index=False)

#time.sleep(20)

#battingDataDate = scraper.parse(page_url="teams/tgl.cgi?team=CHC&t=b&year=2019", date="2019-04-20")
#print(battingDataDate)
#battingDataDate.to_csv('singleRow.csv', index=False)


