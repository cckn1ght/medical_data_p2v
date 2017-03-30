from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from collections import namedtuple
import pandas as pd
import csv


def performance(tweet_data, filtered_tweet_data, tp_file, prediction_file, similarity, add_key):
    pred = [0] * len(tweet_data)
    qualified_ids = set(t.id for t in filtered_tweet_data)
    prediction = list()
    tags = [t.tag for t in tweet_data]
    for num, tweet in enumerate(tweet_data):
        if tweet.id in qualified_ids:
            prediction.append(tweet)
            # print(tweet, raw_sentences[num])
            pred[num] = 1
    if similarity == 0.7 and not add_key:
        with open(prediction_file, 'w') as f:
            writer = csv.writer(f)
            for t in prediction:
                writer.writerow(t.get_row())
    print('similarity: {}, added key: {}'.format(similarity, str(add_key)))
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
    Performance = namedtuple('Performance', [
                             'accuray', 'percision', 'recall', 'f1', 'roc', 'TP', 'FP', 'TN', 'FN'])
    return Performance(accuracy_score(tags, pred), precision_score(tags, pred), recall_score(tags, pred), f1_score(tags, pred), roc_auc_score(tags, pred), tp, fp, tn, fn)
