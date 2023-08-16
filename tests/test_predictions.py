"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentiment_analysis_model import __version__ as _version
from sentiment_analysis_model.config.core import config
from sentiment_analysis_model.predict import make_prediction
from sentiment_analysis_model.processing.features import load_test_dataset


def test_make_prediction(sample_input_data):
    # Given
    test_data = load_test_dataset()
    
    for data in test_data:
        # user_input = "The biscuits did not taste great!"
        user_input = data
        predicted_sentiment, sentiment_probability = make_prediction(user_input)
        print("Predicted Sentiment:", predicted_sentiment)
        print("Sentiment Probability:", sentiment_probability)


def test_precision(sample_input_data):
    # Given
    test_data = load_test_dataset()
    
    for data in test_data:
        # user_input = "The biscuits did not taste great!"
        user_input = data
        predicted_sentiment, sentiment_probability = make_prediction(user_input)
        print("Predicted Sentiment:", predicted_sentiment)
        print("Sentiment Probability:", sentiment_probability)
        
    assert sentiment_probability > 0.8
    