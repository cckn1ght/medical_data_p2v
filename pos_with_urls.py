import load_6546
from load_shared_task import load_shared_task
import re


def load_csv(filename='6546', yes_only=True):
    if filename == '6546':
        ids, raw_sentences, tags = load_6546.load(yes_only=yes_only)
    elif filename == '4819':
        ids, raw_sentences, tags = load_shared_task(yes_only=yes_only)
    return ids, raw_sentences, tags


def pos_with_ursl(filename='6546', yes_only=True):
    ids, raw_sentences, tags = load_csv(filename=filename, yes_only=yes_only)
    sen_with_urls = []
    count = 0
    for sen in raw_sentences:
        urls = re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sen)
        if urls:
            sen_with_urls.append(sen)
            count += 1
    print(count)
    return sen_with_urls

sen_with_urls = pos_with_ursl(filename='6546', yes_only=False)
