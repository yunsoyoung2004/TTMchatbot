# utils.py 또는 drift/utils.py

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

def get_sia():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()
