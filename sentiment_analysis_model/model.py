import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
import json

from config.core import config
from sentiment_analysis_model.processing.features import load_review_data
from sentiment_analysis_model.processing.data_manager import load_tokenizer

# Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,Score,Time,Summary,Text

def create_model(p_optimizer, p_loss, p_metrics):
    EMBEDDING_DIM = 32
    
    l_maxlen = config.model_config.maxlen

    tokenizer= load_tokenizer()
    
    print('Build model...')
    vocab_size = len( tokenizer.word_index) + 1
    
    print ('number of unique words in the corpus:', vocab_size)

    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length= l_maxlen))
    model.add(LSTM(units=40,  dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Try using different optimizers and different optimizer configs
    model.compile(loss=p_loss, optimizer=p_optimizer, metrics=[p_metrics])

    print('Summary of the built model...')
    print(model.summary())   
    
    return model

# main logic
        
# Create model
def get_model_classifier():
    
    classifier = create_model(
                          p_optimizer = config.model_config.optimizer, 
                          p_loss = config.model_config.loss, 
                          p_metrics = [config.model_config.accuracy_metric])
    
    return classifier
                          
