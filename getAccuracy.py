import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def main():

    connection = sqlite3.connect("C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\games\\gamesSchedule.db")
    crsr = connection.cursor()
    sql_command = "SELECT logHomeWinPred, actualHomeWin FROM games"


    predictions = pd.DataFrame(crsr.execute(sql_command), columns=["Predicted", "Actual"])
    #print(predictions.head(10))

    #sql_command = "SELECT logAwayWinPred, actualAwayWin FROM games"

    #awayPredictions = pd.DataFrame(crsr.execute(sql_command), columns=["Predicted", "Actual"])

    #predictions = predictions.append(awayPredictions)
    #print(predictions.dtypes)

    from sklearn.metrics import classification_report
    print(classification_report(predictions['Actual'], predictions['Predicted']))

if __name__ == "__main__":
    main()
