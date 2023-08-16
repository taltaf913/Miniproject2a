import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sentiment_analysis_model.config.core import config
from sentiment_analysis_model.config.core import TRAINED_MODEL_DIR
from sentiment_analysis_model import __version__ 

from sentiment_analysis_model.model import get_model_classifier
from sentiment_analysis_model.processing.data_manager import callbacks_and_save_model
from sentiment_analysis_model.processing.data_manager import get_reviews_dataset, load_tokenizer, handle_tokenize

def run_training() -> None:
    
    """
    Train the model.
    """
    X_train, X_test, y_train, y_test = get_reviews_dataset()    
    tokenizer, X_train_pad = handle_tokenize( X_train)
            
    l_epochs = config.model_config.epochs
    l_batchsize = config.model_config.batch_size
    l_validation_split = config.model_config.validation_size
    
    l_callback_methods = callbacks_and_save_model()
    
    classifier = get_model_classifier()
    
    history = classifier.fit(X_train_pad, y_train, batch_size= l_batchsize, epochs= l_epochs, verbose=1,
                    validation_split= l_validation_split, callbacks= l_callback_methods)

    # Calculate the score/error
    #test_loss, test_acc = classifier.evaluate(test_data)
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)

    
if __name__ == "__main__":
    run_training()