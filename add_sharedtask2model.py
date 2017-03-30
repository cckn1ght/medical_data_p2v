from gensim.models import Phrases, Word2Vec
import re
import json
from nltk.corpus import stopwords
import pprs
stopwords_list = stopwords.words('english')


with open('./sharedTask/tweets.json', 'r') as f:
    more_sentences = []
    for line in f:
        tweet = json.loads(line)
        raw_tweet_text = tweet['text']
        # Discard tweets with URL.
        urls = re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', raw_tweet_text)
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
                more_sentences.append(pprs.string2words(
                    tweet_text_str, remove_stopwords=False))

bigram = Phrases.load('./cleaned_data/bigramPhraseModel_skipG')
trigram = Phrases.load('./cleaned_data/trigramPhraseModel_skipG')

bigram.add_vocab(more_sentences)
trigram.add_vocab(bigram[more_sentences])


bigram.save('./cleaned_data/bigramPhraseModel_skipG')
trigram.save('./cleaned_data/trigramPhraseModel_skipG')

model = Word2Vec.load('./cleaned_data/p2vModel_skipG')
model.train(trigram[bigram[more_sentences]])
model.save('./cleaned_data/p2vModel_skipG_withSharedTask')
