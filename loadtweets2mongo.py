import os
from pymongo import MongoClient
import json


def get_filenames():
    files_list = []
    for folder in os.listdir('/Volumes/MacDisk2/ravish_data/s3'):
        if folder.startswith('instance'):
            for file in os.listdir('/Volumes/MacDisk2/ravish_data/s3/{}'.format(folder)):
                if file.endswith('json'):
                    file_name = '/Volumes/MacDisk2/ravish_data/s3/{}/{}'.format(
                        folder, file)
                    files_list.append(file_name)
    return files_list


def generate_tweet():
    files = get_filenames()
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    tweet = json.loads(line)['text']
                    yield tweet
                except:
                    yield ''


def copy_to_mongo():
    client = MongoClient()
    db = client.ravishData
    collection = db.AWSTweets
    for n, tweet in enumerate(generate_tweet()):
        if tweet == '':
            continue
        tweet_info = {'tweet': tweet}
        collection.insert(tweet_info)
        if n % 1000 == 0:
            print('inserted {} tweets'.format(n))
    print('done')


if __name__ == '__main__':
    copy_to_mongo()
