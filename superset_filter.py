import json
from apply_filter import sentence2phrases, performance, save_prediction
import load_6546
from pprs import string2words
from collections import Counter
from load_shared_task import load_shared_task
from gensim.models.phrases import Phraser, Phrases


def load_phrase_model():
    print('loading phrases model...')
    bigram = Phraser(Phrases.load('./cleaned_data/bigramPhraseModel'))
    trigram = Phraser(Phrases.load('./cleaned_data/trigramPhraseModel'))
    return bigram, trigram


def load_superset_similar_words(similar_words_file, topn=0, similarity=0.7, add_key=False):
    with open(similar_words_file, 'r') as f:
        similarity_dict = json.load(f)
    sim_words_set = set()

    for key, val in similarity_dict.items():
        if topn:
            words = {sim_word[0] for sim_word in val[:topn]}
            if add_key:
                words.add(key)
        if similarity:
            words = {sim_word[0]
                     for sim_word in val if float(sim_word[1]) >= similarity}
            if words and add_key:
                words.add(key)
        if (not topn) and (not similarity):
            words = {sim_word[0] for sim_word in val}
            if add_key:
                words.add(key)
        # words.add(key)
        sim_words_set |= words

    with open('./cleaned_data/similar_words.txt', 'w') as f:
        for word in sim_words_set:
            f.write(word + '\n')
    return sim_words_set


def super_sim_filter(ids, phrase_sentences, tags, topn=1, similarity=0.7, add_key=False):
    print('applying filter...')
    sim_words_set = load_superset_similar_words(
        './cleaned_data/sim_transpose.json', topn=topn, similarity=similarity, add_key=add_key)
    print('There are {} similar words totally'.format(len(sim_words_set)))
    # filter sentences
    qualified_sentences = []
    qualified_tags = []
    qualified_ids = []
    for tweet_id, sentence, tag in zip(ids, phrase_sentences, tags):
        if not set(sentence).isdisjoint(sim_words_set):
            qualified_sentences.append(sentence)
            qualified_tags.append(tag)
            qualified_ids.append(tweet_id)
    print('tweets left after filter: {}'.format(len(qualified_tags)))
    print(Counter(qualified_tags))
    # print('No to Yes ratio after filter: {}'.format(
    #     Counter(qualified_tags)[0] / Counter(qualified_tags)[1]))
    return qualified_ids, qualified_tags


def apply_filter(bigram, trigram, filename='6546', topn=1, similarity=0.7, add_key=False):
    if filename == '6546':
        ids, raw_sentences, tags = load_6546.load(yes_only=False)
        tp_file = './test/Matrika/iteration2/tp_iteration2_superset_cleaned_data.csv'
        fname = './test/Matrika/iteration2/6546_prediction_superset_cleaned_data.csv'
    elif filename == '4819':
        ids, raw_sentences, tags = load_shared_task(yes_only=False)
        tp_file = './test/4819_shared_task/tp_4819_superset_cleaned_data.csv'
        fname = './test/4819_shared_task/4819_prediction_superset_cleaned_data.csv'
    sentences = [string2words(s, remove_stopwords=True)
                 for s in raw_sentences]
    phrase_sentences = sentence2phrases(sentences, bigram, trigram)
    qualified_ids, qualified_tags = super_sim_filter(
        ids, phrase_sentences, tags, topn=topn, similarity=similarity, add_key=add_key)
    print('--------------------------------------')
    id_truetag_predtag = performance(
        ids, qualified_ids, tags, raw_sentences, tp_file)
    save_prediction(fname, id_truetag_predtag, raw_sentences)
    return id_truetag_predtag


def main():
    bigram, trigram = load_phrase_model()
    apply_filter(bigram, trigram, filename='4819',
                 topn=0, similarity=0.8, add_key=True)


if __name__ == '__main__':
    main()
