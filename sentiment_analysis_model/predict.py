import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_analysis_model import __version__ as _version
from sentiment_analysis_model.config.core import config
from sentiment_analysis_model.processing.data_manager import load_model
from sentiment_analysis_model.processing.features import strip_html, remove_punctuations, remove_stopwords, load_test_dataset
from sentiment_analysis_model.processing.data_manager import load_tokenizer

def make_prediction(p_input_text):
    # Preprocess the input text
    preprocessed_text = strip_html(p_input_text)
    preprocessed_text = remove_punctuations(preprocessed_text)
    preprocessed_text = remove_stopwords(preprocessed_text)
    
    l_maxlen = config.model_config.maxlen

    # Tokenize and pad the preprocessed text
    tokenizer = load_tokenizer()
    input_sequence = tokenizer.texts_to_sequences([preprocessed_text])
    input_sequence = pad_sequences(input_sequence, maxlen= l_maxlen)

    # Predict sentiment
    sentiment_probability = model_to_use.predict(input_sequence)[0][0]
    predicted_sentiment = 'positive' if sentiment_probability > 0.5 else 'negative'

    return predicted_sentiment, sentiment_probability

# main load the model
model_file_name = f"{config.app_config.model_save_file}{_version}"
model_to_use = load_model(file_name = model_file_name)

if __name__ == "__main__":
    
    test_data = load_test_dataset()
    
    for data in test_data:
        # user_input = "The biscuits did not taste great!"
        user_input = data
        predicted_sentiment, sentiment_probability = make_prediction(user_input)
        print("Predicted Sentiment:", predicted_sentiment)
        print("Sentiment Probability:", sentiment_probability)