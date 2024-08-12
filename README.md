# Titanic Data Analysis and ML Training

## Project Overview

This project involves analyzing the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic) to explore various factors that might have influenced the survival of passengers. After performing exploratory data analysis (EDA), a Logistic Regression, SVM, KNN and Decision Tree models are trained to predict passenger survival based on key features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is the famous Titanic dataset, which includes data on passengers such as their age, sex, fare, class, and whether they survived. The dataset can be downloaded from the [Kaggle Titanic competition page](https://www.kaggle.com/c/titanic).

### Features

- **PassengerId**: Unique ID for each passenger.
- **Survived**: Survival (0 = No, 1 = Yes).
- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard the Titanic.
- **Parch**: Number of parents/children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Fare paid for the ticket.
- **Cabin**: Cabin number.
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook (optional, but recommended for interactive exploration)

## Exploratory Data Analysis (EDA)

The EDA phase involves visualizing and analyzing the data to identify patterns and relationships between features. Key findings include:

- **Gender and Survival**: Females had a higher survival rate than males.
- **Passenger Class and Survival**: Passengers in 1st class had a higher survival rate.
- **Age Distribution**: Younger passengers, especially children, had a better chance of survival.

Visualizations were created using `matplotlib` and `seaborn`.

## Modeling

I have tried more than one model and watched which is better 

The model's performance was evaluated using:

- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: To visualize the performance of the classification.
- **Precision, Recall, and F1-Score**: For a more detailed analysis of model performance.

## Results

- The Logistic Regression model achieved an accuracy of 77% on the test data.
- The SVM model achieved an accuracy of 76% on the test data.
- The KNN model achieved an accuracy of 72% on the test data.
- The Decision Tree model achieved an accuracy of 78% on the test data.
so, The Decision Tree model is the best

## Conclusion

This project demonstrates how exploratory data analysis and machine learning can be applied to historical datasets to derive insights and make predictions. While The Decision Tree model provided good predictions, further improvement could be achieved by exploring more sophisticated models or additional feature engineering.

## Acknowledgements

- The dataset is provided by Kaggle.
- Inspiration from various public notebooks and tutorials on Kaggle.