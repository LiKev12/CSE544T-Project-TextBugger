from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import requests
import json

def c_google():


    # Instantiates a client
    client = language.LanguageServiceClient()

    # The text to analyze
    text = u'I hate being late.'
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment

    print('Text: {}'.format(text))
    print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))




def c_ibm():

    url = 'http://max-text-sentiment-classifier.max.us-south.containers.appdomain.cloud/model/predict'
    myobj = {
        "text": [
            "Happy"
        ]
    }
    x = requests.post(url, data = json.dumps(myobj))
    print(x)



c_ibm()