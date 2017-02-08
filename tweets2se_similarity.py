from load_shared_task import load_shared_task
from gensim.models import Word2Vec
from p2v_similarity import load_se
from apply_filter import load_phrase_model
from collections import namedtuple
import csv
from pprs import string2words

bigram, trigram = load_phrase_model()
model = Word2Vec.load('./p2vModel/p2v_model')
ids_4819, raw_sentences_4819, tags_4819 = load_shared_task(
    yes_only=False, raw=True)
sentences_4819 = [string2words(s, remove_stopwords=False)
                  for s in raw_sentences_4819]
se_set = load_se()
print('Generating phrases...')
phrase_sentences = list(trigram[bigram[sentences_4819]])


def tweets2se_similarity(sentences, ids, side_effects, model):
    SEsim = namedtuple('SEsim', ['word', 'SE', 'similarity'])
    result = {}
    for tid, words in zip(ids, sentences):
        result[tid] = []
        for word in words:
            max_sim = 0
            max_se = None
            for se in side_effects:
                try:
                    sim = model.similarity(word, se)
                    if sim > max_sim:
                        max_sim = sim
                        max_se = se
                except KeyError:
                    pass
            SS = SEsim(word=word, SE=max_se, similarity=max_sim)
            result[tid].append(SS)
    return result


result = tweets2se_similarity(phrase_sentences, ids_4819, se_set, model)

with open('tweets2se_similarity.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['tweet_id', 'text', 'tag', 'highest similarity', 'lowest similarity', 'average similarity', 'details...'])
    for tid, sentence, tag in zip(ids_4819, raw_sentences_4819, tags_4819):
        total = 0.
        highest = 0.
        lowest = 1.
        for SE in result[tid]:
            total += SE.similarity
            highest = max(highest, SE.similarity)
            lowest = min(highest, SE.similarity)
        average = total / len(result[tid])
        writer.writerow([tid, sentence, tag, highest, lowest, average] + result[tid])
