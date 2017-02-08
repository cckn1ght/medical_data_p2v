import csv
from pprs import string2words
from collections import Counter


def load(yes_only=False):
    ids = []
    sentences = []
    tags = []
    tags_table = {'yes': 1, 'no': 0}
    with open('./test/Matrika/Iteration2/6546.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            tag = tags_table[line[2].lower()]
            if yes_only:
                if tag:
                    ids.append(line[0])
                    sentences.append(line[1])
                    tags.append(tag)
            else:
                ids.append(line[0])
                sentences.append(line[1])
                tags.append(tags_table[line[2].lower()])
    print('loaded 6546 data set, the original data distribution:')
    print(Counter(tags))
    return ids, sentences, tags
