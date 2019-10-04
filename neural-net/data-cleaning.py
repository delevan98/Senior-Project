import pandas as pd
import os


def main():
    #os.chdir('C:\\Users\\Mike Delevan\\git\\Senior-Project\\data-scraper')
    os.chdir('C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\data-scraper')
    extension = 'csv'

    teamAbbr = ["CHN","PHI","PIT", "CIN", "SLN", "BOS", "CHA",
				"CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
			    "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
				"TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
				"MIL", "WAS"]
    #Explore each variable in dataset
    for x in range(30):
        all_filenames = []
        for y in range(2010,2019):
            all_filenames.append(teamAbbr[x] + str(y) + ".csv")
            #print(all_filenames)

        if(os.path.isfile(teamAbbr[x]+ '_All.csv')):
            print("File has already been created!")
        else:
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

        data['Visiting Team Batting Average'] = data['Visting Team Hits'] / (data['Visting Team At-Bats'] - data['Visiting Team Sac Hits'] - data['Visting Team Sac Flys'] - data['Visting Team HBP'] - data['Visting Team Walks'] - data['Visting Team Int Walks'])
        data['Home Team Batting Average'] = data['Home Team Hits'] / (data['Home Team At-Bats'] - data['Home Team Sac Hits'] - data['Home Team Sac Flys'] - data['Home Team HBP'] - data['Home Team Walks'] - data['Home Team Int Walks'])
        data['Visiting Team OBP'] = (data['Visting Team Hits'] + data['Visting Team Walks'] + data['Visting Team HBP']) / (data['Visting Team At-Bats'] + data['Visting Team Walks'] + data['Visting Team HBP'] + data['Visting Team Sac Flys'])
        data['Home Team OBP'] = (data['Home Team Hits'] + data['Home Team Walks'] + data['Home Team HBP']) / (data['Home Team At-Bats'] + data['Home Team Walks'] + data['Home Team HBP'] + data['Home Team Sac Flys'])
        data['Visiting Team Slugging'] = ((data['Visting Team Hits'] - data['Visting Team Doubles'] - data['Visting Team Triples'] - data['Visting Team Home-Runs']) + (data['Visting Team Doubles'] * 2) + (data['Visting Team Triples'] * 3) + (data['Visting Team Home-Runs'] * 4)) / (data['Visting Team At-Bats'])
        data['Home Team Slugging'] = ((data['Home Team Hits'] - data['Home Team Doubles'] - data['Home Team Triples'] - data['Home Team Home-Runs']) + (data['Home Team Doubles'] * 2) + (data['Home Team Triples'] * 3) + (data['Home Team Home-Runs'] * 4)) / (data['Home Team At-Bats'])
        data['Visiting Team OPS'] = data['Visiting Team OBP'] + data['Visiting Team Slugging']
        data['Home Team OPS'] = data['Home Team OBP'] + data['Home Team Slugging']

        data.loc[(data['Home Team'] == teamAbbr[x]) & (data['Home Team Score'] > data['Visiting Team Score']), 'Win'] = 1

        data.loc[(data['Visting Team'] == teamAbbr[x]) & (data['Visiting Team Score'] > data['Home Team Score']), 'Win'] = 1

        data.loc[(data['Home Team'] == teamAbbr[x]) & (data['Home Team Score'] < data['Visiting Team Score']), 'Win'] = 0

        data.loc[(data['Visting Team'] == teamAbbr[x]) & (data['Visiting Team Score'] < data['Home Team Score']), 'Win'] = 0

        data.to_csv(teamAbbr[x] + '_All.csv', index=False)

        final = pd.DataFrame(columns=['teamAbbr', 'League', 'Score', 'isHomeTeam', 'wonPrev', 'atBats', 'Hits',
                                      'Doubles', 'Triples', 'homeRuns', 'RBI', 'Walks', 'Strikeouts', 'LOB',
                                      'pitchersUsed', 'indER', 'teamER', 'Errors', 'battingAverage', 'OBP', 'Slugging',
                                      'OPS', 'Win'])

        data.drop(['Visting Team', 'League', 'Home Team', 'League.1', 'Park ID'], axis=1, inplace=True)
        data.drop(['Winning Pitcher ID', 'Losing Pitcher ID', 'Saving Pitcher ID', 'Visiting Starter Pitcher ID',
                   'Home Starter Pitcher ID'], axis=1, inplace=True)

        fillTeamDF(data, final, teamAbbr[x])
        corr_matrix = data.corr()
        #print(corr_matrix["Home Team Score"].sort_values(ascending=False))
        corr_matrix.to_csv(
            "C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\"+ teamAbbr[x]+"_correlation_matrix.csv")

        description = data.describe()
        description.to_csv("C:\\Users\\Mike Delevan\\PycharmProjects\\Senior-Project\\neural-net\\stats-and-correlations\\"+ teamAbbr[x]+"_desc_stats.csv")

        write_averages(data, teamAbbr[x])

        #print(data['Win'].value_counts())

def write_averages(data, teamAbbr):
    if (os.path.isfile('team_averages.csv')):
        print("File has already been created!")
    else:
        avg_file = open("team_averages.csv", "w+")
        data = dropCols(data)
        data['Team Abbr'] = "Team Abbr"
        data.iloc[:0].to_csv('team_averages.csv', index=False, header=True)

    try:
        data = dropCols(data)
    except KeyError:
        print("Columns have already been deleted!")

    description = data.tail(10).describe()
    print(description)
    description['TeamAbbr'] = teamAbbr
    description.iloc[1:2].to_csv('team_averages.csv', index=False, header=False,mode='a')

def dropCols(data):
    data.drop(['Date', 'Visting Team Stolen Bases', 'Home Team Stolen Bases',
               'Visting Team Caught Stealing',
               'Home Team Caught Stealing', 'Visting Team G Double Play', 'Home Team G Double Play',
               'Visting Team Awarded First on Interference', 'Home Team Awarded First on Interference',
               'Visting Team Balks', 'Home Team Balks', 'Visting Team Put-outs', 'Home Team Put-outs',
               'Visting Team Assists', 'Visting Team Passed Balls', 'Home Team Passed Balls',
               'Visting Team Double Plays', 'Attendance', 'Home Team Double Plays',
               'Home Team Triple Plays', 'Visting Team Triple Plays', 'Home Team Triples',
               'Visiting Team Sac Hits', 'Home Team Int Walks', 'Visting Team Int Walks',
               'Home Team Sac Hits', 'Length of Game in Outs', 'Visting Team Sac Flys', 'Home Team Sac Flys',
               'Home Team Wild Pitches', 'Home Team HBP', 'Visting Team HBP', 'Visting Team Wild Pitches',
               'Visiting Team Game Number', 'Home Team Game Number'], axis=1, inplace=True)

    try:
        data.drop(['Unnamed: 75'], axis=1, inplace=True)

    except KeyError:
        print("Column is not in the file!!!")

    return data

#This function was created in DS 201 on October 4, 2019
def fillTeamDF(data, final, teamAbbr):

    final['teamAbbr'] = teamAbbr
    if(data['Home Team'] == teamAbbr):
        final['League'] = data['League.1']
        final['Score'] = data['Home Team Score']
        final['isHomeTeam'] = 1
        final['wonPrev'] = data['Home Team Score'].shift() > data['Visiting Team Score']
        final['atBats'] = data['Home Team At-Bats']
        final['Hits'] = data['Home Team Hits']
        final['Doubles'] = data['Home Team Doubles']
        final['Triples'] = data['Home Team Triples']
        final['homeRuns'] = data['Home Team Home-Runs']
        final['RBI'] = data['Home Team RBI']
        final['Walks'] = data['Home Team Walks']
        final['Strikeouts'] = data['Home Team Strikeouts']
        final['LOB'] = data['Home Team LOB']
        final['pitchersUsed'] = data['Home Team Pitchers Used']
        final['indER'] = data['Home Team Ind ER']
        final['teamER'] = data['Home Team Team ER']
        final['Errors'] = data['Home Team Erros']
        final['battingAverage'] = data['Home Team Batting Average']
        final['OBP'] = data['Home Team OBP']
        final['Slugging'] = data['Home Team Slugging']
        final['OPS'] = data['Home Team OPS']
        final['Win'] = data['Win']

    else:
        final['League'] = data['League']
        final['Score'] = data['Visiting Team Score']
        final['isHomeTeam'] = 0
        final['wonPrev'] = data['Visiting Team Score'].shift() > data['Home Team Score']
        final['atBats'] = data['Visting Team At-Bats']
        final['Hits'] = data['Visting Team Hits']
        final['Doubles'] = data['Visting Team Doubles']
        final['Triples'] = data['Visting Team Triples']
        final['homeRuns'] = data['Visting Team Home-Runs']
        final['RBI'] = data['Visting Team RBI']
        final['Walks'] = data['Visting Team Walks']
        final['Strikeouts'] = data['Visting Team Strikeouts']
        final['LOB'] = data['Visting Team LOB']
        final['pitchersUsed'] = data['Visting Team Pitchers Used']
        final['indER'] = data['Visting Team Ind ER']
        final['teamER'] = data['Visting Team Team ER']
        final['Errors'] = data['Visting Team Erros']
        final['battingAverage'] = data['Visiting Team Batting Average']
        final['OBP'] = data['Visiting Team OBP']
        final['Slugging'] = data['Visiting Team Slugging']
        final['OPS'] = data['Visiting Team OPS']
        final['Win'] = data['Win']

        print(final.tail(10))






if __name__ == "__main__":
    main()