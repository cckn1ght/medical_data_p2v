from gensim.models import Phrases, Word2Vec
from pymongo import MongoClient
import pprs
import logging
import csv
import json
from itertools import chain
from functools import reduce
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

client = MongoClient('localhost', 27017)
db = client.ravishData
collection = db.tweets


def preprocess(tweetObj):
    if 'text' in tweetObj and 'lang' in tweetObj and tweetObj['lang'] == 'en':
        text = tweetObj['text']
        entities = tweetObj['entities']
        urls = (e["url"] for e in entities["urls"])
        users = ("@" + e["screen_name"] for e in entities["user_mentions"])
        media_urls = ()
        if 'media' in entities:
            media_urls = (e["url"] for e in entities["media"])
        text = reduce(lambda t, s: t.replace(s, ""),
                      chain(urls, media_urls, users), text)
        text = pprs.string2words(text, remove_stopwords=True)
        return text
    else:
        return ['']


def multigen(gen_func):
    class _multigen(object):

        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen


@multigen
def sentence_generator():
    for tweet in collection.find():
        yield preprocess(tweet)
    p = './drug-chatter-and-downloadscript/tweets.json'
    with open(p, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            yield preprocess(tweet)
    # with open('./sharedTask/tweets.json', 'r') as f:
    #     for line in f:
    #         tweet = json.loads(line)
    #         tweet_text = tweet['text']
    #         yield pprs.string2words(tweet_text, remove_stopwords=False)


def phrases_model(sentences, min_count=30, threshold=100):
    bigram_transformer = Phrases(
        sentences, min_count=min_count, threshold=threshold)
    trigram_transformer = Phrases(
        bigram_transformer[sentences], min_count=min_count, threshold=threshold)
    trigram_phrases_set = set()
    for phrase, score in trigram_transformer.export_phrases(bigram_transformer[sentences]):
        phraselist = phrase.decode('utf-8').split(' ')
        ph = '_'.join(phraselist)
        trigram_phrases_set.add(ph)

    with open('./phrases/trigram_phrases_mincount{min}_threshold{thre}.csv'.format(min=min_count, thre=threshold), 'w') as f:
        writer = csv.writer(f)
        for phrase in trigram_phrases_set:
            writer.writerow([phrase])
    bigram_transformer.save('./p2vModel/bigramPhraseModel')
    # trigram = Phraser(trigram_transformer)
    trigram_transformer.save('./p2vModel/trigramPhraseModel')
    return bigram_transformer, trigram_transformer


sentences = sentence_generator()

bigram, trigram = phrases_model(sentences, 30, 100)
model = Word2Vec(trigram[bigram[sentences]],
                 size=200, window=5, min_count=10, workers=4, max_vocab_size=200000)
model.save('./p2vModel/p2vModel')

model.save_word2vec_format(
    './p2vModel/p2v_model_with_vocab', fvocab='./p2vModel/p2v_vocab')
