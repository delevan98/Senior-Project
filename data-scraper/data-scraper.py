from bs4 import BeautifulSoup
import requests

teamAbbr = ["CHN", "PHI", "PIT", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]
url = "http://www.baseball-reference.com/teams/tgl.cgi?team=" + teamAbbr + "&t=b&year=" + str(year)

URL = 'https://www.baseball-reference.com/teams/tgl.cgi?team=CHC&t=b&year=2019#rowsum_desc'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

results = soup.find(id='team_batting_gamelogs')

stats = soup.find_all('td', attrs = {'data-stat': 'RBI'})

print(stats)

