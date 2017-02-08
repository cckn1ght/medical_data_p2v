import json
from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from pprs import string2words


def load_shared_task():
    sentences = []
    with open('./sharedTask/tweets.json', 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweet_text = tweet['text']
            sentences.append(string2words(tweet_text, remove_stopwords=False))
    return sentences


sentences = load_shared_task()

bigram = Phrases.load('./p2vModel/bigramPhraseModel')
trigram = Phrases.load('./p2vModel/trigramPhraseModel')

bigram.add_vocab(sentences)
trigram.add_vocab(bigram[sentences])
