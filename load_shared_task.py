import json
from pprs import string2words
from collections import Counter


def load_shared_task(yes_only=False, raw=True):
    tweet_id_tag = {}
    with open('./sharedTask/Twitter_corpus_releaseset_external.txt', 'r') as f:
        for line in f:
            line = line.split()
            tweet_id_tag[line[0]] = line[2]
    ids = []
    sentences = []
    tags = []
    with open('./sharedTask/tweets.json', 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweet_id = tweet['id_str']
            tweet_text = tweet['text']
            if not raw:
                tweet_text = string2words(tweet_text, remove_stopwords=False)
            tweet_tag = int(tweet_id_tag[tweet_id])
            if (yes_only and tweet_tag) or not yes_only:
                ids.append(tweet_id)
                sentences.append(tweet_text)
                tags.append(tweet_tag)
    print(Counter(tags))
    if not yes_only:
        print('Original No to Yes ratio: {}'.format(
            Counter(tags)[0] / Counter(tags)[1]))
    return ids, sentences, tags
