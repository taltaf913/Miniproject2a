
from bs4 import BeautifulSoup 
from datetime import datetime
import re
import os

# language kit
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# pandas, tensorflow 
import pandas as pd

# internal modules
from sentiment_analysis_model.config.core import config

def strip_html(text):
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
    
def remove_stopwords(text, is_lower_case=False):
    # splitting strings into tokens (list of words)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    updated_stopword_list = create_and_update_stopwords_list()
    
    if is_lower_case:
        # filtering out the stop words
        filtered_tokens = [token for token in tokens if token not in updated_stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in updated_stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text    
    
""" 
This will initialize, review and construct required set of stop-words
"""
def create_and_update_stopwords_list():
    # setting english stopwords
    stopword_list = nltk.corpus.stopwords.words('english')
    updated_stopword_list = []

    for word in stopword_list:
        if word=='not' or word.endswith("n't"):
           pass
        else:
           updated_stopword_list.append(word)

    # print(updated_stopword_list)    
    return updated_stopword_list
        
# removing punctuations
def remove_punctuations(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text

def handle_cleanup_text_data(p_reviews):
    
    l_column_names = config.app_config.clean_up_words
    
    elements = l_column_names.split(",")
    for e in elements:
        l_columnname= e.strip()
        p_reviews[l_columnname] = p_reviews[l_columnname].apply(strip_html)
        p_reviews[l_columnname] = p_reviews[l_columnname].apply(remove_punctuations)
        p_reviews[l_columnname] = p_reviews[l_columnname].apply(remove_stopwords)
    
    return p_reviews

def handle_unwanted_columns(p_reviews):
    print("remove_unwanted_columns() reviews_shape: ", p_reviews.shape)
    features = config.app_config.not_required_features
    elements = features.split(",")
    for e in elements:
        e= e.strip()
        print("removing freature: ", e)
        p_reviews.pop( e)
    return p_reviews
        
def handle_duplicates(p_reviews):
    # cols_to_check_for_duplicates = ["ProductId","ProfileName", "Text"]
    cols_to_check_for_duplicates = ["Sentiment", "Text"]
    # l_dups = reviews.duplicated( subset=cols_to_check_for_duplicates, keep="first")
    # l_dups = reviews.duplicated( subset=["Sentiment"], keep="first")
    # l_dups = p_reviews.duplicated( subset=["Text"], keep=False)
    # l_reviews = p_reviews.drop_duplicates( subset=["Text"], keep="first")
    l_reviews = p_reviews.drop_duplicates( subset=cols_to_check_for_duplicates, keep="first")
    
    return l_reviews

def handle_append_newcolumns(p_reviews):
    # date_time = pd.to_datetime( reviews.pop('Time'), format='%d.%m.%Y %H:%M:%S')
    df_time = p_reviews["Time"].apply( datetime.fromtimestamp)
    
    # adding Sentiment column according to available Score
    # reviews['Sentiment'] = reviews['Score'].apply(lambda x: "positive" if x>=3 else "negative")
    p_reviews['Sentiment'] = p_reviews['Score'].apply(lambda x: 1 if x>=3 else 0)
    p_reviews["Time"] = df_time
    
    return p_reviews

def read_data_from_file():
    reviews = pd.read_csv( config.app_config.dataset_file_path)
    
    return reviews

# test data, TODO: load from a file
def load_test_dataset():
     test_data = [
         "biscuit did not taste good",
         "cold idly tastes better",
         "vanilla flavoured creamy biscuits are very popular with kids"
     ] 
     
     return test_data

# Performing the data augmentation as series of transformations
def load_review_data():
    
    # load from file
    # stop words
    # clean/remove columns
    # duplicates
    # append columns

    l_reviews_data = read_data_from_file()
    
    # l_reviews_data = handle_cleanup_text_data("Text")
    l_reviews_data = handle_cleanup_text_data( l_reviews_data)
    
    l_reviews_data = handle_unwanted_columns(l_reviews_data)
    l_reviews_data = handle_append_newcolumns(l_reviews_data)
    # doing in this sequence because we are checking duplicates on default-column "Text" and added-column "Sentiment"
    l_reviews_data = handle_duplicates(l_reviews_data) # Output: 393579 after removing Text alone, with Sentiment+Text it is 393591
 
    return l_reviews_data


# reviews_data = get_reviews_data()
