from gensim.models import Word2Vec
import load_6546
from apply_filter import load_phrase_model, sentence2phrases
import build_super_set
import queue as q
import json
from pprs import string2words
import csv


class SimilaryWordPair(object):

    def __init__(self, tweet_word, se_word, similarity):
        self.tweet_word = tweet_word
        self.se_word = se_word
        self.similarity = similarity

    def __lt__(self, other):
        return self.similarity < other.similarity

    def get_values(self):
        return self.se_word, self.similarity


def main():
    bigram, trigram = load_phrase_model()
    w2v_model = Word2Vec.load('./p2vModel/p2v_model')
    ids, raw_sentences, tags = load_6546.load()
    sentences = [string2words(s, remove_stopwords=True)
                 for s in raw_sentences]
    phrase_sentences = sentence2phrases(sentences, bigram, trigram)
    super_se_set, super_se_set_cui = build_super_set.build_set(True)
    sim_dict = built_sim_dict(
        phrase_sentences, super_se_set, w2v_model, topn=5)
    # deep_simiwords(phrase_sentences, ids, tags, sim_dict)
    with open('sub_module_sim_words.json', 'w') as f:
        json.dump(sim_dict, f, indent=2)

    tweet_detail = []
    for tid, raw_sent, tag, phrase_sent in zip(ids, raw_sentences, tags, phrase_sentences):
        word_sim_pair = [(w, *sim_dict[w][-1])
                         for w in phrase_sent if sim_dict[w]]
        sim_lambda = lambda a: a[-1]
        max_sim = 0
        min_sim = 0
        average_sim = 0
        if word_sim_pair:
            max_sim = max(word_sim_pair, key=sim_lambda)[-1]
            min_sim = min(word_sim_pair, key=sim_lambda)[-1]
            average_sim = sum(
                [sim for a, b, sim in word_sim_pair]) / len(word_sim_pair)
        tweet_detail.append([tid, raw_sent, tag, max_sim,
                             min_sim, average_sim] + word_sim_pair)

    with open('sub_model_sim_words.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['tweet_id', 'raw_tweet', 'tag',
                         'max_similarity', 'min_similarity', 'average_similarity'])
        for row in tweet_detail:
            writer.writerow(row)

    with open('sub_model_features.csv', 'w') as f:
        writer = csv.writer(f)
        for row in tweet_detail:
            li = [round(row[3], 3), round(row[4], 3), round(row[5], 3), row[2]]
            if li[-1] == 0:
                li[-1] = 'no'
            else:
                li[-1] = 'yes'
            writer.writerow(li)


def built_sim_dict(phrase_sentences, super_se_set, w2v_model, topn=5):
    print("building similar words dictionary")
    all_words_set = set()
    for sent in phrase_sentences:
        for word in sent:
            all_words_set.add(word)
    sim_dict = {}
    for word in all_words_set:
        sim_priorityQ = q.PriorityQueue(topn)
        for se in super_se_set:
            try:
                similarity = round(w2v_model.similarity(word, se), 3)
                sim_word_pair = SimilaryWordPair(word, se, similarity)
                if sim_priorityQ.full():
                    sim_priorityQ.get()
                sim_priorityQ.put(sim_word_pair)
            except KeyError:
                pass
        li = []
        while not sim_priorityQ.empty():
            li.append(sim_priorityQ.get().get_values())
        sim_dict[word] = li[:]
    return sim_dict


main()
# def deep_simiwords(phrase_sentences, ids, tags, sim_dict):
#     for sen in phrase_sentences:
#         for word in sen:
