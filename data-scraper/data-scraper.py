from bs4 import BeautifulSoup
import requests

URL = 'https://www.baseball-reference.com/teams/tgl.cgi?team=CHC&t=b&year=2019#rowsum_desc'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

results = soup.find(id='team_batting_gamelogs')

#games = results.find_all('section', class_='left')

print(results.prettify())

