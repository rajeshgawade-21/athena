import pandas as pd
import re
import emoji
import nltk
from datetime import datetime
from google.cloud import language_v1
import os
from google.cloud.language_v1 import enums
from google.cloud.language_v1 import types
import apache_beam as beam
from apache_beam.ml.gcp import naturallanguageml as nlp
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Rajesh/python/folkloric-alpha-314903-a9dfcf2018b2.json"

nltk.download('words')
words = set(nltk.corpus.words.words())
path =  'c:/Rajesh/python/file_original_21.csv'
path_1 = 'c:/Rajesh/python/file_clean_21.csv'
trump_df = pd.read_csv(path)

from google.cloud import language_v1

def getClassification(text_content):
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    response = client.classify_text(request = {'document': document})
    # Loop through classified categories returned from the API
    for category in response.categories:
        # Get the name of the category representing the document.
        # See the predefined taxonomy of categories:
        # https://cloud.google.com/natural-language/docs/categories
        print(u"Category name: {}".format(category.name))
        # Get the confidence. Number representing how certain the classifier
        # is that this category represents the provided text.
        #print(u"Confidence: {}".format(category.confidence))
    return category.name


def sample_analyze_sentiment(text_content):
    """
    Analyzing Sentiment in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'I am so happy and joyful.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Get overall sentiment of the input document
    print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    print(
        u"Document sentiment magnitude: {}".format(
            response.document_sentiment.magnitude
        )
    )
    # Get sentiment for all sentences in the document
    for sentence in response.sentences:
        print(u"Sentence text: {}".format(sentence.text.content))
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    print(u"Language of the text: {}".format(response.language))

def getSentiment(text_content):
    client = language_v1.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    return response.document_sentiment.score

def getMagnitude(text_content):
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}
    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    return response.document_sentiment.magnitude

def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    #tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    #     if w.lower() in words or not w.isalpha())
    return tweet

trump_df['text'] = trump_df['text'].map(lambda x: cleaner(x))
print("Job Start")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
trump_df['clean_date'] = pd.to_datetime(trump_df['tweetcreatedts']).dt.date
#trump_df['sentiment']=trump_df['text'].map(lambda x: getSentiment(x))
#trump_df['magnitude']=trump_df['text'].map(lambda x: getMagnitude(x))
#Classification works only when tweets are minimum 20 words
#trump_df['category']=trump_df['text'].map(lambda x: getClassification(x))
trump_df.to_csv(path_1)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print("Job Done")