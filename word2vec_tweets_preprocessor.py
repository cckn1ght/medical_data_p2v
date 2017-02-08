# -*- coding: utf-8 -*-
import re
from pymongo import MongoClient

import logging
from nltk.corpus import stopwords

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

client = MongoClient('localhost', 27017)
db = client.ravishData
collection = db.AWSTweets

stopwords_list = stopwords.words('english')


def get_tweets():
    cleaned_tweets = ""
    count = 0
    for tcount, tweet in enumerate(collection.find()):
        raw_tweet_text = tweet['tweet']
        # Discard tweets with URL.
        urls = re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', raw_tweet_text)
        if tcount % 10000 == 0:
            print('dealing with No {} tweets'.format(tcount))
        if not urls:
            # Strip User Mentions.
            tweet_text = ' '.join(re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", raw_tweet_text).split())
            # Strip Stopwords.
            no_stop_list = [
                word for word in tweet_text.lower().split() if word not in stopwords_list]
            tweet_text = ' '.join(no_stop_list)
            # Strip if tweet have less than 2 words.
            if len(tweet_text.split()) > 2:
                # Keep only len(words) >= 4.
                tweet_text = [w for w in tweet_text.split() if len(w) >= 4]
                tweet_text_str = " ".join(tweet_text)
                cleaned_tweets = cleaned_tweets + " " + tweet_text_str
                count += 1
    print('total tweets after cleaning: ', count)
    text_file = open("cleaned_data.txt", "a")
    text_file.write(cleaned_tweets)
    text_file.close()

get_tweets()
