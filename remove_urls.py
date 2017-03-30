import re
from nltk.corpus import stopwords
from Annotation import Annotation


stopwords_list = stopwords.words('english')


def remove_urls(annotations):

    cleaned_annotations = list()
    for tweet in annotations:
        raw_tweet_text = tweet.text
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
                cleaned_annotations.append(Annotation(
                    tweet.id, tweet_text_str, tweet.tag))
    return cleaned_annotations
