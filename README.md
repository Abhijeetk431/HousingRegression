# HousingRegression

## Introduction
This is a project created as part of MLOps Assignment 1. In this project the core task is to design, implement, and automate a machine learning workflow for predicting house prices using the Boston Housing dataset. It will include use of classical machine learning models, their performance comparision, and then enhancing it by including hyperparameter tuning. Automation will be handled using GitHub Actions.

## Setup
Clone this repo on your local machine
`git clone git@github.com:Abhijeetk431/HousingRegression.git`

Change directory to use the git repo
`cd HousingRegression`

Once in the directory, intialize conda environment using commands
`conda init`
`conda create -n housingregression python=3.13`
`conda activate housingregression`

This will acoda environment that the project will use, in case some changes to requirements is required, edit requirements.in and append the required package in the end of the file. After that compile the requirements using command
`pip install pip-tools`
`pip-compile requirements.in`

This will update requirements.txt file with the latest required packages. Post this run the below command to install the dependencies
`pip install -r requirements.txt`

## Data pull
Once the setup is ready, the code to pull data is in utils.py. To run the code, just use the below commands.
`python` <- This will open a Python Interpreter
`from utils import load_data`
`data = laod_data()`
`print(data)`

## Regression Models
This project uses Linear ,Decision Tree and Random Forest regression models. The pulled data is split into test and train data sets. Train data is used to fit the model, and then, the model is used to derive target values which are compared to test data. By comparision, we can derive the values of MSE and R2. To run regression models fitting and performance evaluation, run
`python regression.py`
Ensure that the code is checkout out to reg_branch, else the main branch with hyperparamter tuning will also run

## Hyper parameter tuning
Since linear regression does not support atleast 3 hyperparametrs, for hyperparameter tuning, this project gets rid of linear regression model and in placeof it introduces Ridge regression model. To run the model fitting and performance evaluation along with hyperparameter tuning, from main or hyper_branch in the repo run command
`python regression.py`

## Conclusion
Thus, this project covers ask of using three different Regression ML models for predicting the house prices using Boston Housing Dataset along with ability of hyperparameter tuning.