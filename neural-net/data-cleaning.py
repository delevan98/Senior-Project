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

        data.drop(['Visting Team', 'League', 'Home Team', 'League.1', 'Park ID'], axis=1, inplace=True)
        data.drop(['Winning Pitcher ID', 'Losing Pitcher ID', 'Saving Pitcher ID', 'Visiting Starter Pitcher ID',
                   'Home Starter Pitcher ID'], axis=1, inplace=True)

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
if __name__ == "__main__":
    main()