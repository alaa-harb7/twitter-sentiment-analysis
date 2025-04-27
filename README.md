#  Twitter Sentiment Analysis

## 1- Overview
The **Twitter Entity Sentiment Analysis** project focuses on analyzing sentiments expressed in tweets related to various entities.  
This project uses several machine learning models to classify sentiments into predefined categories and aims to identify patterns in public opinions on Twitter.

## 2- Objectives
- Perform sentiment analysis on Twitter data.
- Classify tweets into Positive, Negative, Neutral, and Irrelevant categories.
- Evaluate and compare different machine learning models to find the best performing one.
- Build a FastAPI web application for real-time sentiment prediction.

## 3- Technologies & Tools
- **Python**
- **Spacy**: For data preprocessing
- **FastAPI**: For building the web app backend.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Joblib**: For model serialization and loading.
- **gdown**: To download the trained models.
- **HTML + Jinja2**: For frontend templating and displaying results dynamically.
- **Uvicorn**: For running the FastAPI server.

## 4- Data Overview
The project uses a **Twitter Sentiment Dataset**, labeled into the following classes:
- **Positive** 😊
- **Negative** 😞
- **Neutral** 😐
- **Irrelevant** ❓

Each tweet is classified into one of these sentiment categories based on its content.

## 5- Models Used
The following machine learning algorithms were trained and evaluated:
- Logistic Regression
- Random Forest Classifier ✅
- AdaBoost Classifier
- XGBoost Classifier
- CatBoost Classifier

> **Note:**  
> The **Random Forest Classifier** achieved the highest accuracy among all tested models and was selected for deployment.

## 6- Web Application
- The app is built using **FastAPI** and allows users to input tweet text.
- It instantly predicts the sentiment and displays the result beautifully with dynamic UI.
- Includes **Dark Mode** toggle and animated result cards.

## 🔗 Links
- **Kaggle Notebook**: [kaggle](https://www.kaggle.com/code/alaaharb7/sentiment-analysis-twitter)
- **GitHub Repository**: [GitHub Repo](https://github.com/alaa-harb7/twitter-sentiment-analysis)
