[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
# Predicting the Outcomes of Baseball Games Using Machine Learning Approaches

The app is hosted at https://machinelearningbaseball.herokuapp.com/ for your viewing pleasure.

# Overview

This project aims to predict the outcomes of baseball games using machine learning approaches AKA POBGUMLA. The predictions that will be made are the winner of the game (Win/Loss) and the score for both the home team and the away team. The project will collect historical data from Retrosheet.com and scrape data from Baseball-Reference.com. This data will then be cleaned, processed, and feature engineered to wrangle data into a form that will enhance machine learning model performance. Once this step has been completed the data will be fed into regression and classification models to train and test them. Once the highest performing regression and classification models have been chosen, they will be saved and loaded into a Flask web-app. This web app will have two separate pages each containing their own functionality. The game matchup page will use the saved models to predict the winner and the respective scores for games that are fed into it. It will then present its predictions in a user friendly way, like ESPN. The other page will allow a user to input their own matchups through the use of a dropdown menu. This hypothetical matchup will then be fed into the models and a winner will be predicted along with the respective scores.

# Languages/Frameworks
* Python
* Flask
* Bootstrap CSS
* Jinja
* Javascript
* Scikit-learn
* Tensorflow
* Jupyter Notebook

# Requirements
I've had a lot of trouble in the past using requirements.txt files therefore one is not included. Usually when I cloned project from Github and tried to install the packages using the requirements.txt file, there were always errors or version incompatibilities. If there is a way to make a requirements.txt file without specifying the version number please let me know. For now here is a list of the library requirements:

* pandas
* os
* matplotlib
* matplotlib.pyplot
* numpy
* sklearn
* seaborn
* pickle
* tensorflow
* flask

# Running the  project
Once the project has been completely cloned and all requirements have been installed, all you have to do is run server.py to access the web-app. There are plans in the future to host the web-app on heroku so that people won't have to clone the project to access the web-app.

If you want to run the project from the beginning, start with data-cleaning.py. However, I do have to warn you to delete the team_averages.csv file before you do so. Part of the data-cleaning.py file appends a team's 10 game average to a csv file. If the file is not deleted the averages will keep getting appended without deleting the old ones, leading to a very large file.

After succesfully running data-cleaning.py the next file is ready to be executed, neural-network.py. Once that file has finished running you are now ready to run the server.py file!


# TO-DO
- [x] Data Pre-proccessing
- [x] Simple machine learning models
- [x] Simple web-app
- [x] Get 2019 MLB Schedule
- [x] Hypothetical matchup page
- [x] Dark mode
- [ ] Prediction checks and balances
- [ ] Calculate additional sabermetrics
- [ ] Convert abbreviations to real names
- [ ] Allow users to see predictions for multiple days
- [ ] Create web-scraper/ Get 2019 data
- [ ] Get and save predictions for 2019 games
- [ ] Better models (ex. Neural networks)
- [ ] Incorporate pitching data
