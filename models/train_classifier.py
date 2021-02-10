"""
Training the classifier to identify categories based on a given message
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Arguments:
    1) Path to SQLite destination database (e.g. disaster_response.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""
#import relevant libraries and modules
import sys
import os
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import multioutput
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """
    Load Data from SQLlite
    
    Arguments:
        database_filepath -> Path to SQLite destination database (disaster_response.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM messages",engine)
    #removing related=2 as this might be erroneous
    df = df[df["related"] != 2]
    # storing features (messages) in X
    X = df["message"] 
    # Storing all the labels (categories) in Y
    Y = df.iloc[:,4:]
    # Storing category names
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize a given text
    
    Arguments:
        text -> Text message 
    Output:
        clean_tokens -> List of tokens extracted from the text
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    clean_tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens if word not in stop_words]
    
    return clean_tokens    
    pass


def build_model():
    """
    Build Pipeline a machine learning pipeline function
    
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', multioutput.MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame (Y_pred, columns = Y_test.columns)
    print(Y_test.describe(), Y_pred.describe())
    for column in Y_test.columns:
        print('Model accuracy for: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))



def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()