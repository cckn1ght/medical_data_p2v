import json
import csv
from pprs import string2words
from collections import Counter
from gensim.models.phrases import Phraser, Phrases
from p2v_similarity import extract_sim, load_se
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import pandas as pd
from load_shared_task import load_shared_task
import load_6546


def load_similar_words(similar_words_file):
    with open(similar_words_file, 'r') as f:
        similarity_dict = json.load(f)
    sim_words_set = set()

    for key, val in similarity_dict.items():
        words = [sim_word[0] for sim_word in val]
        words.append(key)
        sim_words_set |= set(words)

    with open('similar_words.txt', 'w') as f:
        for word in sim_words_set:
            f.write(word + '\n')
    return sim_words_set


def load_superset_similar_words(similar_words_file, topn=None):
    with open(similar_words_file, 'r') as f:
        similarity_dict = json.load(f)
    sim_words_set = set()

    for key, val in similarity_dict.items():
        words = set()
        if topn:
            for i, sim_word in enumerate(val):
                if i > topn - 1:
                    break
                words.add(sim_word[0])
        else:
            words = {sim_word[0] for sim_word in val}
        words.add(key)
        sim_words_set |= words

    with open('./superSetModel/similar_words.txt', 'w') as f:
        for word in sim_words_set:
            f.write(word + '\n')
    return sim_words_set


def load_data():
    sentences = []
    tags = []
    tag_lookup = {'no': 0, 'yes': 1}
    with open('./8770+821tweets/New_TrainingSet_8770.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            sentence = ' '.join(line[10:])
            words = string2words(sentence, remove_stopwords=False)
            sentences.append(words)
            tags.append(tag_lookup[line[0].lower()])
    return sentences, tags


def load_phrase_model():
    print('loading phrases model...')
    bigram = Phraser(Phrases.load('./p2vModel/bigramPhraseModel'))
    trigram = Phraser(Phrases.load('./p2vModel/trigramPhraseModel'))
    return bigram, trigram


def sentence2phrases(sentences, bigram, trigram):
    phrase_sentences = list(trigram[bigram[sentences]])
    print('applied phrases model to our sentences')
    return phrase_sentences


def sim_filter(ids, phrase_sentences, tags, similarity=None):
    print('applying filter...')
    if similarity:
        sim_words_set = load_similar_words(
            './phrases/p2v_phrase_similarity>{:d}%.json'.format(int(similarity * 100)))
    else:
        sim_words_set = load_similar_words(
            './phrases/p2v_phrase_similarity.json')
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
# sentences, tags = load_data()
# phrase_sentences = sentence2phrases(sentences)


def performance(ids, qualified_ids, tags, raw_sentences, tp_file):
    pred = [0] * len(ids)
    for num, tid in enumerate(ids):
        if tid in qualified_ids:
            pred[num] = 1
    print('accuracy socre: {}'.format(accuracy_score(tags, pred)))
    print('precision socre: {}'.format(precision_score(tags, pred)))
    print('recall socre: {}'.format(recall_score(tags, pred)))
    print('f1 socre: {}'.format(f1_score(tags, pred)))
    print('roc socre: {}'.format(roc_auc_score(tags, pred)))
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for pre, true in zip(pred, tags):
        if pre == 1 and true == 1:
            tp += 1
        elif pre == 1 and true == 0:
            fp += 1
        elif pre == 0 and true == 1:
            fn += 1
        else:
            tn += 1
    d = {'predicted yes': pd.Series([tp, fp], index=[
        'yes', 'no']), 'predicted no': pd.Series([fn, tn], index=['yes', 'no'])}
    df = pd.DataFrame(d, columns=['predicted yes', 'predicted no'])
    print('Confusion Matrix: ')
    print(df)
    print('------------------------------------')
    id_truetag_predtag = []

    # true positive examples
    # with open(tp_file, 'w') as f:
    #     writer = csv.writer(f)
    #     for pred_tag, true_tag, tid, raw_s in zip(pred, tags, ids, raw_sentences):
    #         id_truetag_predtag.append((tid, true_tag, str(pred_tag)))
    #         if str(true_tag) == str(pred_tag) == '1':
    #             sim_set = extract_sim_words(0.9, raw_s)
    #             writer.writerow([tid, raw_s, sim_set])
    return id_truetag_predtag


def extract_sim_words(similarity, raw_s):
    sim_words_set = load_similar_words(
        './phrases/p2v_phrase_similarity>{:d}%.json'.format(int(similarity * 100)))
    sent = string2words(raw_s, remove_stopwords=False)
    phrase = sentence2phrases(sent, bigram, trigram)
    sim_words = [w for w in phrase if w in sim_words_set]
    return sim_words


def load_essential():
    model = Word2Vec.load('./p2vModel/p2v_model')
    side_effects_set = load_se()
    return model, side_effects_set


def save_prediction(fname, id_truetag_predtag, sentences):
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['tweet_id', 'tweet_text', 'true_tag', 'pred_tag'])
        for (tid, true_tag, pred_tag), sent in zip(id_truetag_predtag, sentences):
            writer.writerow(
                [str(tid), str(sent), str(true_tag), str(pred_tag)])
        print(fname + ' file saved.')


def apply_filter_to_6546(topn, similarity=0.8, eval_sim=True):
    ids, raw_sentences, tags = load_6546.load(yes_only=False)
    sentences = [string2words(s, remove_stopwords=False)
                 for s in raw_sentences]
    phrase_sentences = sentence2phrases(sentences, bigram, trigram)
    extract_sim(model, side_effects_set, topn, similarity)
    if eval_sim:
        qualified_ids, qualified_tags = sim_filter(
            ids, phrase_sentences, tags, similarity)
    else:
        qualified_ids, qualified_tags = sim_filter(
            ids, phrase_sentences, tags)
    print('--------------------------------------')
    tp_file = './test/Matrika/iteration2/tp_iteration2.csv'
    id_truetag_predtag = performance(
        ids, qualified_ids, tags, raw_sentences, tp_file)
    fname = './test/Matrika/iteration2/6546_yes_only_prediction_top{}_sim{}%.csv'.format(
        topn, similarity * 100)
    save_prediction(fname, id_truetag_predtag, raw_sentences)
    return id_truetag_predtag


def apply_filter_to_4819(topn, similarity=0.8, eval_sim=True):
    ids, raw_sentences, tags = load_shared_task(yes_only=False)
    sentences = [string2words(s, remove_stopwords=False)
                 for s in raw_sentences]
    phrase_sentences = sentence2phrases(sentences, bigram, trigram)
    extract_sim(model, side_effects_set, topn, similarity)
    if eval_sim:
        qualified_ids, qualified_tags = sim_filter(
            ids, phrase_sentences, tags, similarity)
    else:
        qualified_ids, qualified_tags = sim_filter(
            ids, phrase_sentences, tags)
    print('--------------------------------------')
    tp_file = './test/4819_shared_task/tp_4819_shared_task.csv'
    id_truetag_predtag = performance(
        ids, qualified_ids, tags, raw_sentences, tp_file)
    fname = './test/4819_shared_task/4819_without_self_prediction_top{}_sim{}%.csv'.format(
        topn, similarity * 100)
    save_prediction(fname, id_truetag_predtag, raw_sentences)
    return id_truetag_predtag


def main():
    model, side_effects_set = load_essential()
    ids, sentences, tags = load_shared_task()
    phrase_sentences = sentence2phrases(sentences, bigram, trigram)
    similarity = 0.9
    topn = 30
    id_truetag_predtag = evaluate_pred(
        topn, model, similarity, phrase_sentences, False)


if __name__ == '__main__':
    model, side_effects_set = load_essential()
    bigram, trigram = load_phrase_model()
    id_truetag_predtag = apply_filter_to_4819(30, 0.9, eval_sim=True)
    id_truetag_predtag = apply_filter_to_6546(30, 0.9, eval_sim=True)
