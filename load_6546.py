import csv
from pprs import string2words
from collections import Counter
from Annotation import Annotation


def load(yes_only=False):
    annotations = []
    tags_table = {'yes': 1, 'no': 0}
    with open('./test/Matrika/Iteration2/6546.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            tag = tags_table[line[2].lower()]
            if yes_only:
                if tag:
                    tweet = Annotation(line[0], line[1], tag)
                    # ids.append(line[0])
                    # sentences.append(line[1])
                    # tags.append(tag)
            else:
                tweet = Annotation(
                    line[0], line[1], tags_table[line[2].lower()])
            annotations.append(tweet)
            # ids.append(line[0])
            # sentences.append(line[1])
            # tags.append(tags_table[line[2].lower()])
    print('loaded 6546 data set, the original data distribution:')
    print(Counter(t.tag for t in annotations))
    return annotations
