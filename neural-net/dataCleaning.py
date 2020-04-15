import pandas as pd
import os


def main():
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\baseData')
    #os.chdir('/data-scraper')

    teamAbbr = ["CHN","PHI","PIT", "CIN", "SLN", "BOS", "CHA",
				"CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
			    "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
				"TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
				"MIL", "WAS"]

    #Explore each variable in dataset

    averages = pd.DataFrame(columns=['Score', 'isHomeTeam', 'atBats', 'Hits',
                                     'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                     'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                                     'OPS', 'Win', 'wonPrev', 'WHIP', 'KPercent', 'BBPercent', 'FIP', 'BABIP', 'ERA', 'HAllowed',
                                     'defensiveSO'])
    for x in range(30):
        os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\baseData')
        all_filenames = []
        for y in range(2010,2019):
            all_filenames.append(teamAbbr[x] + str(y) + ".csv")
            #print(all_filenames)

        if(os.path.isfile(teamAbbr[x]+ '_All.csv')):
            os.remove(teamAbbr[x] + "_All.csv")

        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        combined_csv.to_csv(teamAbbr[x]+ "_All.csv", index=False, encoding='utf-8-sig')

        try:
            data = pd.read_csv(teamAbbr[x] + '_All.csv')
            data.drop(['Visiting Team Line Numbers', 'Home Team Line Numbers'],axis=1, inplace=True)
            data.to_csv(teamAbbr[x] + '_All.csv', index=False)

        except KeyError:
            print("These columns are not in the file!")


        data.loc[(data['Home Team'] == "MIA"), 'Home Team'] = "FLO"
        data.loc[(data['Visting Team'] == "MIA"), 'Visting Team'] = "FLO"


        data['Visiting Team Batting Average'] = data['Visting Team Hits'] / (data['Visting Team At-Bats'])
        data['Home Team Batting Average'] = data['Home Team Hits'] / (data['Home Team At-Bats'])
        data['Visiting Team OBP'] = (data['Visting Team Hits'] + data['Visting Team Walks'] + data['Visting Team HBP']) / (data['Visting Team At-Bats'] + data['Visting Team Walks'] + data['Visting Team HBP'] + data['Visting Team Sac Flys'])
        data['Home Team OBP'] = (data['Home Team Hits'] + data['Home Team Walks'] + data['Home Team HBP']) / (data['Home Team At-Bats'] + data['Home Team Walks'] + data['Home Team HBP'] + data['Home Team Sac Flys'])
        data['Visiting Team Slugging'] = ((data['Visting Team Hits'] - data['Visting Team Doubles'] - data['Visting Team Triples'] - data['Visting Team Home-Runs']) + (data['Visting Team Doubles'] * 2) + (data['Visting Team Triples'] * 3) + (data['Visting Team Home-Runs'] * 4)) / (data['Visting Team At-Bats'])
        data['Home Team Slugging'] = ((data['Home Team Hits'] - data['Home Team Doubles'] - data['Home Team Triples'] - data['Home Team Home-Runs']) + (data['Home Team Doubles'] * 2) + (data['Home Team Triples'] * 3) + (data['Home Team Home-Runs'] * 4)) / (data['Home Team At-Bats'])

        data['Visiting Team OPS'] = data['Visiting Team OBP'] + data['Visiting Team Slugging']
        data['Home Team OPS'] = data['Home Team OBP'] + data['Home Team Slugging']

        #data['Visiting Team FP'] = (data['Visting Team Put-outs'] + data['Visting Team Assists']) / (data['Visting Team Put-outs'] + data['Visting Team Assists'] + data['Visting Team Errors'])
        #data['Home Team FP'] = (data['Home Team Put-outs'] + data['Home Team Assists']) / (data['Home Team Put-outs'] + data['Home Team Assists'] + data['Home Team Errors'])


        #Determining if teamAbb[x] won the game
        data.loc[(data['Home Team'] == teamAbbr[x]) & (data['Home Team Score'] > data['Visiting Team Score']), 'Win'] = 1

        data.loc[(data['Visting Team'] == teamAbbr[x]) & (data['Visiting Team Score'] > data['Home Team Score']), 'Win'] = 1

        data.loc[(data['Home Team'] == teamAbbr[x]) & (data['Home Team Score'] < data['Visiting Team Score']), 'Win'] = 0

        data.loc[(data['Visting Team'] == teamAbbr[x]) & (data['Visiting Team Score'] < data['Home Team Score']), 'Win'] = 0

        # Determining if teamAbbr[x] won their previous game
        data.loc[(data['Home Team'].shift() == teamAbbr[x]) & (data['Home Team Score'].shift() > data['Visiting Team Score'].shift()), 'wonPrev'] = 1
        data.loc[(data['Home Team'].shift() == teamAbbr[x]) & (data['Home Team Score'].shift() < data['Visiting Team Score'].shift()), 'wonPrev'] = 0

        data.loc[(data['Visting Team'].shift() == teamAbbr[x]) & (data['Visiting Team Score'].shift() > data['Home Team Score'].shift()), 'wonPrev'] = 1
        data.loc[(data['Visting Team'].shift() == teamAbbr[x]) & (data['Visiting Team Score'].shift() < data['Home Team Score'].shift()), 'wonPrev'] = 0


        # Calculating advanced pitcher statistics
        data['WHIP'] = (data['BB'] + data['H']) / data['IP']
        data['KPercent'] = data['SO'] / data['batters_faced']
        data['BBPercent'] = data['BB'] / data['batters_faced']
        #data['FIP'] = (((13*data['HR'])+(3*(data['BB']+data['HBP']))-(2*data['SO']))/ data['IP']) + getFIPConstant(data['Date'])
        data['BABIP'] = (data['H'] - data['HR']) / (data['AB'] - data['HR'] - data['SO'] + data['SF'])

        data.to_csv(teamAbbr[x] + '_All.csv', index=False)


        final = pd.DataFrame(columns=['teamAbbr', 'League', 'Score', 'isHomeTeam', 'atBats', 'Hits',
                                      'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                      'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                                      'OPS', 'Win', 'wonPrev', 'WHIP',  'KPercent', 'BBPercent', 'FIP', 'BABIP', 'ERA', 'HAllowed', 'defensiveSO'])

        fillTeamDF(data, final, teamAbbr[x])
        averages = averages.append(write_averages(teamAbbr[x]))
        print('In main')
        print(averages)

    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    averages.to_csv('team_averages.csv', index=False)


def write_averages(teamAbbr):

    os.chdir("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper")
    data = pd.read_csv(teamAbbr + '_Full.csv')

    data.drop(['League', 'teamAbbr'], axis=1, inplace=True)

    data[['Score', 'isHomeTeam', 'atBats', 'Hits', 'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB', 'pitchersUsed', 'indER', 'teamER', 'Errors', 'HAllowed', 'defensiveSO']] = data[['Score', 'isHomeTeam', 'atBats', 'Hits', 'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB', 'pitchersUsed', 'indER', 'teamER', 'Errors', 'HAllowed', 'defensiveSO']].astype(int)

    description = data.tail(10).describe()
    description['TeamAbbr'] = teamAbbr

    lastGame = data.tail(1)
    averageRow = description.iloc[1]

    averageRow['wonPrev'] = lastGame.iloc[0]['wonPrev']

    return averageRow

#This function was created in DS 201 on October 4, 2019
def fillTeamDF(data, final, teamAbbr):

    for (idx, row) in data.iterrows():
        #print(row.loc['Home Team'])
        #print(idx)

        constant = getFIPConstant(row['Date'])
        fip = (((13 * row['HR']) + (3 * (row['BB'] + row['HBP'])) - (2 * row['SO'])) / row['IP']) + constant

        if(row.loc['Home Team'] == teamAbbr):
            final.loc[idx+1] = [teamAbbr, row['League.1'], row['Home Team Score'], 1, row['Home Team At-Bats'], row['Home Team Hits'], row['Home Team Doubles'],
                               row['Home Team Triples'], row['Home Team Home-Runs'], row['Home Team RBI'], row['Home Team Walks'], row['Home Team Strikeouts'],
                               row['Home Team LOB'], row['Home Team Pitchers Used'], row['Home Team Ind ER'], row['Home Team Team ER'], row['Home Team Errors'],
                               row['Home Team Batting Average'], row['Home Team OBP'], row['Home Team Slugging'], row['Home Team OPS'], row['Win'], row['wonPrev'],
                               row['WHIP'],  row['KPercent'], row['BBPercent'], fip, row['BABIP'], row['earned_run_avg'], row['H'], row['SO']]

        else:
            final.loc[idx+1] = [teamAbbr, row['League'], row['Visiting Team Score'], 0, row['Visting Team At-Bats'],
                              row['Visting Team Hits'], row['Visting Team Doubles'],
                              row['Visting Team Triples'], row['Visting Team Home-Runs'], row['Visting Team RBI'],
                              row['Visting Team Walks'], row['Visting Team Strikeouts'],
                              row['Visiting Team LOB'], row['Visting Team Pitchers Used'], row['Visting Team Ind ER'],
                              row['Visting Team Team ER'], row['Visting Team Errors'],
                              row['Visiting Team Batting Average'], row['Visiting Team OBP'], row['Visiting Team Slugging'],
                              row['Visiting Team OPS'], row['Win'], row['wonPrev'], row['WHIP'],  row['KPercent'],
                              row['BBPercent'], fip, row['BABIP'], row['earned_run_avg'], row['H'], row['SO']]

    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    final.to_csv(teamAbbr + '_Full.csv', index=False)

def getFIPConstant(date):
    constants = {
        "2010": 3.079,
        "2011": 3.025,
        "2012": 3.095,
        "2013": 3.048,
        "2014": 3.132,
        "2015": 3.134,
        "2016": 3.147,
        "2017": 3.158,
        "2018": 3.161,
    }

    gameDate = str(date)
    fipConstant = constants[gameDate[0:4]]

    return fipConstant


if __name__ == "__main__":
    main()