from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import re

teamAbbr = ["CHN", "PHI", "PIT", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]

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

        def findDate(dateToSearch, rowToSearch):
            searchString = dateToSearch + '\.[A-Z]+[0-9]+'
            print(searchString)
            regex = re.compile(searchString)
            match = regex.findall(rowToSearch)
            print(match)

            return match[0]


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

        ################################################
        if(date != ""):
            rows = results.find('tbody').find_all('tr')
            print(rows)
        else:
            rows = results.find('tbody').find_all('tr')

        x = 1
        ## Need to figure out how to search for a date in a row and include only that dates data ##
        ## if else above may no longer be needed ^^^^ ##
        for row in rows:
           data = row.find_all("td")
           data_list = []

           for stat in data:
               #if(date != "" and findDate(date, stat) != ""):
         ###################################################
               #print(stat)
               ## 1 denotes a home game, in b-r it is either nothing or an @ ##
               if stat.string == None:
                   data_list.append(1)
               else:
                   data_list.append(str(stat.get_text()))

           gameData = dict(zip(header_names, data_list))
           print(gameData)
           ## If the dictionary is not empty add it to the dataframe ##
           if bool(gameData) != False:
                df.loc[x] = gameData
                x = x + 1
           else:
                print("Found header in middle of table!")


        return df


scraper = BaseballScraper()
#battingData = scraper.parse(page_url="teams/tgl.cgi?team=CHC&t=b&year=2019")
#print(battingData)
#battingData.to_csv('batting.csv', index=False)

#time.sleep(20)

#pitchingData = scraper.parse(page_url="teams/tgl.cgi?team=CHC&t=p&year=2019")
#print(pitchingData)
#pitchingData.to_csv('pitching.csv', index=False)

#time.sleep(20)

battingDataDate = scraper.parse(page_url="teams/tgl.cgi?team=CHC&t=b&year=2019", date="2019-04-20")
print(battingDataDate)

