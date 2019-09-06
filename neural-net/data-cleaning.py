import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import sys
import csv

def main():
    os.chdir('C:\\Users\\Mike Delevan\\git\\Senior-Project\\data-scraper')
    extension = 'csv'

    teamAbbr = ["CHN","PHI","PIT", "CIN", "SLN", "BOS", "CHA",
				"CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
			    "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
				"TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
				"MIL", "WAS"]

    for x in range(30):
        all_filenames = []
        for y in range(2010,2019):
            all_filenames.append(teamAbbr[x] + str(y) + ".csv")
            print(all_filenames)

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



        data['Visiting Team Batting Average'] = data['Visting Team Hits'] / (data['Visting Team At-Bats'] - data['Visiting Team Sac Hits'] - data['Visting Team Sac Flys'] - data['Visting Team HBP'] - data['Visting Team Walks'] - data['Visting Team Int Walks'])
        data['Home Team Batting Average'] = data['Home Team Hits'] / (data['Home Team At-Bats'] - data['Home Team Sac Hits'] - data['Home Team Sac Flys'] - data['Home Team HBP'] - data['Home Team Walks'] - data['Home Team Int Walks'])
        data['Visiting Team OBP'] = (data['Visting Team Hits'] + data['Visting Team Walks'] + data['Visting Team HBP']) / (data['Visting Team At-Bats'] + data['Visting Team Walks'] + data['Visting Team HBP'] + data['Visting Team Sac Flys'])
        data['Home Team OBP'] = (data['Home Team Hits'] + data['Home Team Walks'] + data['Home Team HBP']) / (data['Home Team At-Bats'] + data['Home Team Walks'] + data['Home Team HBP'] + data['Home Team Sac Flys'])
        data['Visiting Team Slugging'] = ((data['Visting Team Hits'] - data['Visting Team Doubles'] - data['Visting Team Triples'] - data['Visting Team Home-Runs']) + (data['Visting Team Doubles'] * 2) + (data['Visting Team Triples'] * 3) + (data['Visting Team Home-Runs'] * 4)) / (data['Visting Team At-Bats'])
        data['Home Team Slugging'] = ((data['Home Team Hits'] - data['Home Team Doubles'] - data['Home Team Triples'] - data['Home Team Home-Runs']) + (data['Home Team Doubles'] * 2) + (data['Home Team Triples'] * 3) + (data['Home Team Home-Runs'] * 4)) / (data['Home Team At-Bats'])
        data['Visiting Team OPS'] = data['Visiting Team OBP'] + data['Visiting Team Slugging']
        data['Home Team OPS'] = data['Home Team OBP'] + data['Home Team Slugging']
        data.to_csv(teamAbbr[x] + '_All.csv', index=False)

        #Make win loss column
        #Do descriptive statistics for each column and put into diff file
        #Run simple machine learning methods (linear regression) to predict win
        #Run heatmaps for variables
        #Use step AIC ()



    data.plot(kind="scatter", x="Home Team OBP", y="Home Team Score")
    plt.show()
    data["Home Team HBP"].hist()
    plt.show()

    corr_matrix = data.corr()
    corr_matrix["Home Team Score"].sort_values(ascending=False)

if __name__ == "__main__":
    main()