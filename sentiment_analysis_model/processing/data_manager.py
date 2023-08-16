import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path
import os

import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

# internal modules
from sentiment_analysis_model.config.core import config
from sentiment_analysis_model import __version__ 
from sentiment_analysis_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from sentiment_analysis_model.processing.features import load_review_data

# default
#
DATA_LOADED="NO"

# Tokenization utilities      
def save_tokenizer(tokenizer_to_save):
    # in same folder where model is saved    
    tokenizer_filename = f"{config.app_config.tokenization_save_file}{__version__}.json"
    # l_save_path = TRAINED_MODEL_DIR / tokenizer_filename
    l_save_path =  tokenizer_filename
    
    print("saving tokenizer to file: "+ str(l_save_path))
    
    # Convert tokenizer to JSON string
    tokenizer_json = tokenizer_to_save.to_json()

    # Save the JSON string to a file
    with open( str(l_save_path), 'w') as json_file:
        json_file.write(tokenizer_json)
        
    json_file.close()
    
    
def load_tokenizer():
    
    tokenizer_filename = f"{config.app_config.tokenization_save_file}{__version__}.json"
    # l_save_path = TRAINED_MODEL_DIR / tokenizer_filename
    l_save_path =  tokenizer_filename
    
    print("\n===================  \n")
    print("\nloading tokenizer: " + str(l_save_path))
    print("\npresent working directory: "+ os.getcwd())
    print("\n===================\n")
    
    if os.path.exists( l_save_path):
    
       # Load the tokenizer from the saved JSON file
       with open(l_save_path, 'r') as json_file:
            loaded_tokenizer_json = json_file.read()

       return tokenizer_from_json(loaded_tokenizer_json)
    else:
        print("\n\ntokenizer file: ",tokenizer_filename, " --- is not found in current folder: ", os.getcwd())
        
   # TODO: if file-not-exists

def handle_tokenize(p_X_train):
    
    p_usesaved_tokenizer = config.app_config.load_existing_tokenizer
    print("handle_tokenize: p_usesaved_tokenizer: ", p_usesaved_tokenizer)    
    
    loaded_tokenizer= False
    if p_usesaved_tokenizer:
        tokenizer = load_tokenizer()          
    # If the file does-not exist then create the tokenizer   
    
    if tokenizer is None:
        print("handle_tokenize: created Tokenizer and fitting-text")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(p_X_train)
        
        # always save tokenizer?
        if config.app_config.save_tokenizer:
            print("handle_tokenize: save_tokenizer: ", save_tokenizer, " -- saving tokenizer")    
            save_tokenizer( tokenizer)
    
    l_maxlen = config.model_config.maxlen
    
    X_train_tok = tokenizer.texts_to_sequences(p_X_train)

    X_train_pad = pad_sequences(X_train_tok, padding='post', maxlen= l_maxlen, truncating='post')            
    
    
    return tokenizer, X_train_pad

# Define a function to return a commmonly used callback_list
def callbacks_and_save_model():
    callback_list = []
    
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{__version__}"
    l_save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep = [save_file_name])

    # Default callback
    
    if config.model_config.earlystop >0:
        early_stopping = EarlyStopping(monitor=config.model_config.monitor, verbose=1, patience=2, min_delta=0.001)
        callback_list.append( early_stopping)
    
    best_model_checkpoint= ModelCheckpoint(filepath = str(l_save_path),
                    save_best_only = config.model_config.save_best_only,
                    verbose=1,
                    monitor = config.model_config.monitor)    
    callback_list.append( best_model_checkpoint)
    
    return callback_list

def load_model(*, file_name: str) -> keras.models.Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = keras.models.load_model(filepath = file_path)
    return trained_model


def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


# main logic

def get_reviews_dataset():
    
    DATA_LOADED = os.environ.get("DATA_LOADED", "NO")
    print("checking if data-is loaded: ", DATA_LOADED)
    # if DATA_LOADED=="NO":
    reviews_data = load_review_data()
    X_train, X_test, y_train, y_test = train_test_split( reviews_data['Text'].values, reviews_data['Sentiment'].values,
                                                    test_size= config.model_config.test_size,
                                                    random_state=config.model_config.random_state)
    os.environ["DATA_LOADED"]="YES"
        
    return X_train, X_test, y_train, y_test

