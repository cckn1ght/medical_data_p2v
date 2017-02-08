from load_shared_task import load_shared_task
from p2v_similarity import load_se
from collections import Counter
from gensim.models.phrases import Phraser
from apply_filter import performance

def sider_filter(ids, phrase_sentences, tags, terminologies_set):
    qualified_sentences = []
    qualified_tags = []
    qualified_ids = []
    for tweet_id, sentence, tag in zip(ids, phrase_sentences, tags):
        if not set(sentence).isdisjoint(terminologies_set):
            qualified_sentences.append(sentence)
            qualified_tags.append(tag)
            qualified_ids.append(tweet_id)
    print('tweets left after filter: {}'.format(len(qualified_tags)))
    print(Counter(qualified_tags))
    print('No to Yes ratio after filter: {}'.format(
        Counter(qualified_tags)[0] / Counter(qualified_tags)[1]))
    return qualified_ids, qualified_tags


def sentence2phrases(sentences):
    bigram = Phraser.load('./p2vModel/bigramPhraseModel')
    trigram = Phraser.load('./p2vModel/trigramPhraseModel')
    phrase_sentences = list(trigram[bigram[sentences]])
    return phrase_sentences


terminologies_set = load_se()
ids, sentences, tags = load_shared_task()
phrase_sentences = sentence2phrases(sentences)
qualified_ids, qualified_tags = sider_filter(
    ids, phrase_sentences, tags, terminologies_set)
pred = performance(ids, qualified_ids, tags)